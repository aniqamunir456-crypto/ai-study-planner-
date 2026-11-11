import json
import os
from datetime import date, datetime, timedelta
from typing import List, Literal

import pandas as pd
import streamlit as st
from groq import Groq
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="StudyFlow AI | Adaptive Study Planner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_PLANNING_DAYS = 120
MAX_DAILY_HOURS = 16


# ---------------------------------------------------------------------
# Pydantic schemas for validated LLM output
# ---------------------------------------------------------------------
class StudyTask(BaseModel):
    date: date
    subject: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    duration_hours: float = Field(gt=0, le=MAX_DAILY_HOURS)
    task_type: Literal[
        "Learning", "Weak Topic", "Revision", "Practice Test", "Break / Buffer"
    ]
    notes: str = Field(default="", max_length=500)

    @field_validator("duration_hours")
    @classmethod
    def round_duration(cls, value: float) -> float:
        return round(value, 2)


class StudyPlan(BaseModel):
    schedule: List[StudyTask] = Field(min_length=1)
    strategy: str = Field(min_length=1)
    study_tips: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#1d2340; --muted:#778199; --line:#e7eaf3; --purple:#6d5ce8; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      background:
        radial-gradient(circle at 90% 0%, rgba(129,105,255,.12), transparent 26rem),
        radial-gradient(circle at 0% 35%, rgba(71,160,213,.07), transparent 25rem),
        #f7f8fc;
      color:var(--ink);
    }
    [data-testid="stHeader"] { background:rgba(247,248,252,.82); }
    [data-testid="stSidebar"] { background:#17182d; border-right:1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] * { color:#eef0ff !important; }
    [data-testid="stSidebar"] .stCaption { color:#b8bdd8 !important; }
    .hero {
      padding:1.65rem 1.85rem; border-radius:24px; color:white;
      background:linear-gradient(118deg,#20213f 0%,#45418e 55%,#785be0 100%);
      box-shadow:0 18px 44px rgba(43,39,104,.18);
      position:relative; overflow:hidden; margin-bottom:1.25rem;
    }
    .hero:after { content:"✦"; position:absolute; right:7%; top:-46px; font-size:170px; color:rgba(255,255,255,.08); }
    .hero-kicker { color:#d9d5ff; font-size:.75rem; font-weight:700; letter-spacing:.15em; }
    .hero h1 { color:white; font:800 clamp(2rem,4vw,2.8rem)/1.12 'Manrope',sans-serif; margin:.4rem 0 .55rem; }
    .hero p { color:#e7e5ff; margin:0; max-width:700px; }
    .section-title { font:800 1.12rem 'Manrope',sans-serif; color:#222743; margin:.25rem 0 .5rem; }
    .muted { color:#7c869d; font-size:.88rem; }
    .surface { background:white; border:1px solid var(--line); border-radius:17px; padding:1rem 1.15rem; box-shadow:0 8px 25px rgba(31,41,75,.035); }
    .step { background:#fff; border:1px solid var(--line); border-radius:15px; padding:.85rem 1rem; }
    .step-label { color:#808aa0; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; }
    .step-value { color:#29264f; font:800 1.2rem 'Manrope',sans-serif; }
    .stButton > button[kind="primary"] { background:linear-gradient(100deg,#6658e8,#8b5cf6); border:0; border-radius:11px; font-weight:700; min-height:2.7rem; box-shadow:0 7px 16px rgba(102,88,232,.18); }
    .stButton > button { border-radius:11px; }
    .stTextArea textarea, .stTextInput input { border-radius:12px !important; border-color:#dfe3ef !important; background:#fdfdff !important; }
    div[data-testid="stExpander"] { border:1px solid #e8eaf3; border-radius:13px; background:#fff; }
    .pill { display:inline-block; background:#f0edff; color:#5c4bd0; padding:.2rem .55rem; border-radius:20px; font-size:.72rem; font-weight:700; }
    .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; font-weight:700; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
def initialize_state():
    defaults = {
        "plan_data": None,
        "plan_inputs": None,
        "completed_tasks": [],
        "plan_created_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ---------------------------------------------------------------------
# API and input helpers
# ---------------------------------------------------------------------
def get_api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret = ""
    return str(secret or os.getenv("GROQ_API_KEY", "")).strip()


def parse_subjects(raw_text: str, today: date) -> list[dict]:
    """
    Expected one subject per line:
    Subject name | YYYY-MM-DD | weak topic 1, weak topic 2
    """
    subjects = []
    seen = set()
    errors = []

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            errors.append(
                f"Line {line_number}: use Subject name | YYYY-MM-DD | weak topics."
            )
            continue

        name = parts[0]
        try:
            exam_date = date.fromisoformat(parts[1])
        except ValueError:
            errors.append(f"Line {line_number}: '{parts[1]}' is not a valid YYYY-MM-DD date.")
            continue

        if exam_date <= today:
            errors.append(f"Line {line_number}: the exam date for {name} must be after today.")
        if name.casefold() in seen:
            errors.append(f"Line {line_number}: subject '{name}' is listed more than once.")
        seen.add(name.casefold())

        weak_topics = []
        if len(parts) == 3 and parts[2]:
            weak_topics = [topic.strip() for topic in parts[2].split(",") if topic.strip()]

        subjects.append({
            "name": name,
            "exam_date": exam_date,
            "days_remaining": (exam_date - today).days,
            "weak_topics": weak_topics,
        })

    if not subjects:
        errors.append("Add at least one subject.")
    if errors:
        raise ValueError("\n".join(errors))
    return subjects


def build_prompt(
    subjects: list[dict],
    daily_hours: float,
    allowed_days: list[str],
    rest_days: list[str],
    start_date: date,
    end_date: date,
) -> str:
    subject_payload = [
        {
            "name": item["name"],
            "exam_date": item["exam_date"].isoformat(),
            "days_remaining_from_today": item["days_remaining"],
            "weak_topics": item["weak_topics"],
        }
        for item in subjects
    ]
    schema = StudyPlan.model_json_schema()
    return f"""
Create a personalized day-by-day study plan. Return ONLY valid JSON matching this schema:
{json.dumps(schema, ensure_ascii=False)}

TODAY: {start_date.isoformat()}
PLANNING WINDOW ENDS: {end_date.isoformat()}
SUBJECTS AND EXAMS:
{json.dumps(subject_payload, ensure_ascii=False, indent=2)}

STUDENT CONSTRAINTS:
- Maximum available study time per study day: {daily_hours} hours.
- Preferred study days: {allowed_days if allowed_days else "Any day except listed rest days"}.
- Rest days (do not schedule any study tasks): {rest_days if rest_days else "None"}.
- Schedule dates must be from tomorrow through the day before the relevant subject's exam.
- Calculate days remaining accurately from TODAY for every exam.
- Never schedule a task for a subject on or after that subject's exam date.
- Use only the exact subject names supplied above.

SCHEDULING RULES:
1. Create a practical schedule with dates and task durations in hours.
2. Give extra time and repeated practice to each subject's weak topics. Explicitly use task_type "Weak Topic" for focused weak-topic sessions.
3. Sequence learning before revision, and revision before practice tests as each exam approaches.
4. Include both "Revision" and "Practice Test" sessions for every subject when its remaining time allows. For very near exams, prioritize high-value revision and at least one practice test if feasible.
5. Allocate more frequent review and practice to the nearest upcoming exams without abandoning later subjects.
6. The sum of all task durations on any one date must not exceed {daily_hours} hours.
7. Do not schedule on rest days or outside preferred study days.
8. Use task_type only from: Learning, Weak Topic, Revision, Practice Test, Break / Buffer.
9. Each task must include date (YYYY-MM-DD), subject, topic, duration_hours, task_type, and notes.
10. Keep sessions focused and realistic (usually 0.5–2 hours each). Split a long day into multiple tasks if useful.
11. Do not create empty schedule days or add tasks beyond the planning window.
12. In "strategy", briefly explain how time was prioritized. In "study_tips", give 3–6 practical study habits.
13. Return JSON only, without markdown fences or extra commentary.
""".strip()


def generate_plan(
    subjects: list[dict],
    daily_hours: float,
    allowed_days: list[str],
    rest_days: list[str],
    today: date,
) -> StudyPlan:
    end_date = max(item["exam_date"] for item in subjects) - timedelta(days=1)
    prompt = build_prompt(
        subjects, daily_hours, allowed_days, rest_days, today + timedelta(days=1), end_date
    )
    client = Groq(api_key=get_api_key())
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic study planner. Respect every date and time constraint. "
                    "Return only a valid JSON object that matches the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
        max_tokens=6000,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The AI returned an empty response. Please try again.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The AI response was not valid JSON. Please try again.") from exc
    return StudyPlan.model_validate(payload)


def validate_plan(
    plan: StudyPlan,
    subjects: list[dict],
    daily_hours: float,
    allowed_days: list[str],
    rest_days: list[str],
    today: date,
) -> list[str]:
    issues = []
    exam_dates = {item["name"].casefold(): item["exam_date"] for item in subjects}
    weak_topics = {
        item["name"].casefold(): [topic.casefold() for topic in item["weak_topics"]]
        for item in subjects
    }
    daily_totals = {}
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for task in plan.schedule:
        subject_key = task.subject.casefold()
        if subject_key not in exam_dates:
            issues.append(f"Unknown subject in task: {task.subject}.")
            continue

        if task.date <= today:
            issues.append(f"Task for {task.subject} is not scheduled in the future ({task.date}).")
        if task.date >= exam_dates[subject_key]:
            issues.append(f"Task for {task.subject} is on or after its exam date ({task.date}).")

        weekday = weekday_names[task.date.weekday()]
        if weekday in rest_days:
            issues.append(f"Task scheduled on a rest day: {weekday}, {task.date}.")
        if allowed_days and weekday not in allowed_days:
            issues.append(f"Task scheduled outside preferred study days: {weekday}, {task.date}.")

        daily_totals[task.date] = daily_totals.get(task.date, 0.0) + task.duration_hours

        if task.task_type == "Weak Topic":
            supplied = weak_topics.get(subject_key, [])
            if supplied and not any(topic in task.topic.casefold() for topic in supplied):
                # This is a warning rather than a hard failure; topic wording may be paraphrased.
                pass

    for task_date, total in daily_totals.items():
        if total > daily_hours + 0.01:
            issues.append(
                f"Daily limit exceeded on {task_date}: {total:.2f} hours scheduled, "
                f"but the limit is {daily_hours:.2f} hours."
            )

    if not plan.schedule:
        issues.append("The AI did not return any study tasks.")
    return issues


def task_dataframe() -> pd.DataFrame:
    plan = st.session_state.plan_data
    if not plan:
        return pd.DataFrame()
    rows = []
    for index, task in enumerate(plan["schedule"]):
        task_id = f"{task['date']}|{task['subject']}|{task['topic']}|{index}"
        rows.append({
            "task_id": task_id,
            "Date": task["date"],
            "Day": date.fromisoformat(task["date"]).strftime("%A"),
            "Subject": task["subject"],
            "Topic / Task": task["topic"],
            "Type": task["task_type"],
            "Hours": task["duration_hours"],
            "Notes": task["notes"],
            "Completed": task_id in st.session_state.completed_tasks,
        })
    return pd.DataFrame(rows)


def export_csv(df: pd.DataFrame) -> bytes:
    export_df = df.drop(columns=["task_id"], errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8-sig")


def export_txt(df: pd.DataFrame) -> str:
    lines = [
        "STUDYFLOW AI — PERSONAL STUDY PLAN",
        "=" * 42,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for day, group in df.sort_values(["Date", "Subject"]).groupby("Date"):
        total_hours = group["Hours"].sum()
        lines.append(f"\n{day} ({group.iloc[0]['Day']}) — {total_hours:.1f} hours")
        lines.append("-" * 42)
        for _, row in group.iterrows():
            status = "DONE" if row["Completed"] else "TODO"
            lines.append(
                f"[{status}] {row['Subject']} | {row['Topic / Task']} | "
                f"{row['Type']} | {row['Hours']} hours"
            )
            if row["Notes"]:
                lines.append(f"      Notes: {row['Notes']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 StudyFlow AI")
    st.caption("Plan smarter. Study with direction.")
    st.divider()
    st.markdown("### Your study workflow")
    st.markdown(
        """
        **01 · Set your goals**  
        Add subjects, exams, and weak topics.

        **02 · Generate your plan**  
        Get a date-aware adaptive timetable.

        **03 · Track your progress**  
        Check off tasks as you complete them.

        **04 · Review and export**  
        Download your study plan anytime.
        """
    )
    st.divider()
    st.markdown("**AI engine**")
    st.code(MODEL_NAME, language="text")
    st.caption("Groq API · Pydantic-validated JSON")


# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">FOCUS BETTER · PREPARE SMARTER</div>
      <h1>Your exams. Your pace.<br>A plan that works.</h1>
      <p>Build a realistic study timetable around your exam dates, available hours, weak topics, and rest days — then track every session.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------
today = date.today()

if st.session_state.plan_data is None:
    st.markdown('<div class="section-title">01 / Set up your study plan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Enter one subject per line using the format below. Exam dates should be later than today.</div>',
        unsafe_allow_html=True,
    )

    example_date_1 = (today + timedelta(days=20)).isoformat()
    example_date_2 = (today + timedelta(days=35)).isoformat()
    example_text = (
        f"Mathematics | {example_date_1} | Integration, Trigonometry\n"
        f"Physics | {example_date_2} | Electromagnetism, Waves\n"
        f"Computer Science | {example_date_2} | Data Structures, Algorithms"
    )

    with st.form("study_plan_form"):
        subjects_text = st.text_area(
            "Subjects, exam dates, and weak topics",
            value=example_text,
            height=155,
            help="Format: Subject name | YYYY-MM-DD | weak topic 1, weak topic 2",
        )

        c1, c2 = st.columns(2)
        with c1:
            daily_hours = st.number_input(
                "Daily available study hours",
                min_value=0.5,
                max_value=float(MAX_DAILY_HOURS),
                value=4.0,
                step=0.5,
            )
            preferred_days = st.multiselect(
                "Preferred study days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                help="Leave all days selected if you have no specific preference.",
            )
        with c2:
            rest_days = st.multiselect(
                "Rest days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Sunday"],
                help="No study tasks will be scheduled on these days.",
            )
            st.markdown(
                '<div class="muted">Rest days override preferred study days. Choose at least one available study day.</div>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "✨ Generate my adaptive study plan",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            subjects = parse_subjects(subjects_text, today)
            overlap = set(preferred_days).intersection(rest_days)
            effective_days = [day for day in preferred_days if day not in rest_days]
            if overlap:
                st.info(
                    "Rest days take priority over preferred study days: "
                    + ", ".join(sorted(overlap))
                    + "."
                )
            if not effective_days:
                raise ValueError("Select at least one study day that is not a rest day.")

            latest_exam = max(item["exam_date"] for item in subjects)
            planning_days = (latest_exam - today).days
            if planning_days > MAX_PLANNING_DAYS:
                raise ValueError(
                    f"The latest exam is {planning_days} days away. "
                    f"For a reliable timetable, keep the planning window within {MAX_PLANNING_DAYS} days."
                )

            if not get_api_key():
                raise ValueError(
                    "GROQ_API_KEY is missing. Add it to Streamlit Cloud → App settings → Secrets."
                )

            with st.spinner("Calculating exam timelines and building your adaptive timetable..."):
                generated = generate_plan(
                    subjects,
                    float(daily_hours),
                    effective_days,
                    rest_days,
                    today,
                )
                validation_issues = validate_plan(
                    generated,
                    subjects,
                    float(daily_hours),
                    effective_days,
                    rest_days,
                    today,
                )
                if validation_issues:
                    raise ValueError(
                        "The generated timetable did not satisfy all constraints:\n- "
                        + "\n- ".join(validation_issues[:12])
                        + "\n\nPlease try generating again, or adjust your study days and available hours."
                    )

                st.session_state.plan_data = generated.model_dump(mode="json")
                st.session_state.plan_inputs = {
                    "subjects": [
                        {
                            "name": item["name"],
                            "exam_date": item["exam_date"].isoformat(),
                            "days_remaining": item["days_remaining"],
                            "weak_topics": item["weak_topics"],
                        }
                        for item in subjects
                    ],
                    "daily_hours": float(daily_hours),
                    "preferred_days": effective_days,
                    "rest_days": rest_days,
                }
                st.session_state.completed_tasks = []
                st.session_state.plan_created_at = datetime.now().isoformat(timespec="seconds")
                st.rerun()

        except ValidationError as exc:
            st.error("The AI response did not match the required timetable structure.")
            with st.expander("Technical details"):
                st.code(str(exc))
        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------
# Generated timetable and progress workspace
# ---------------------------------------------------------------------
if st.session_state.plan_data is not None:
    plan = st.session_state.plan_data
    inputs = st.session_state.plan_inputs
    df = task_dataframe()

    total_tasks = len(df)
    completed_count = int(df["Completed"].sum()) if total_tasks else 0
    progress = completed_count / total_tasks if total_tasks else 0
    total_hours = float(df["Hours"].sum()) if total_tasks else 0.0

    st.markdown('<div class="section-title">02 / Your adaptive study dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Your generated plan is saved in this session. Check tasks off as you complete them.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Study tasks", total_tasks)
    with k2:
        st.metric("Completed", f"{completed_count}/{total_tasks}")
    with k3:
        st.metric("Planned hours", f"{total_hours:.1f} h")
    with k4:
        st.metric("Completion", f"{progress * 100:.0f}%")

    st.progress(progress, text=f"{completed_count} of {total_tasks} study tasks completed")

    st.markdown("### Exam countdown")
    countdown_cols = st.columns(min(len(inputs["subjects"]), 3))
    for index, subject in enumerate(inputs["subjects"]):
        with countdown_cols[index % len(countdown_cols)]:
            exam_date = date.fromisoformat(subject["exam_date"])
            remaining = (exam_date - today).days
            st.markdown(
                f'<div class="step"><div class="step-label">{subject["name"]}</div>'
                f'<div class="step-value">{remaining} days left</div>'
                f'<div class="muted">Exam: {exam_date.strftime("%d %b %Y")}</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("View your planning strategy and study tips", expanded=True):
        st.write(plan["strategy"])
        for tip in plan.get("study_tips", []):
            st.markdown(f"- {tip}")

    st.markdown("### 03 / Daily schedule")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        subject_options = ["All subjects"] + sorted(df["Subject"].unique().tolist())
        subject_filter = st.selectbox("Filter by subject", subject_options)
    with filter_col2:
        type_options = ["All task types"] + sorted(df["Type"].unique().tolist())
        type_filter = st.selectbox("Filter by task type", type_options)
    with filter_col3:
        status_filter = st.selectbox(
            "Completion status",
            ["All tasks", "Pending only", "Completed only"],
        )

    filtered = df.copy()
    if subject_filter != "All subjects":
        filtered = filtered[filtered["Subject"] == subject_filter]
    if type_filter != "All task types":
        filtered = filtered[filtered["Type"] == type_filter]
    if status_filter == "Pending only":
        filtered = filtered[~filtered["Completed"]]
    elif status_filter == "Completed only":
        filtered = filtered[filtered["Completed"]]

    if filtered.empty:
        st.info("No tasks match these filters.")
    else:
        visible_dates = sorted(filtered["Date"].unique())
        for schedule_date in visible_dates:
            day_df = filtered[filtered["Date"] == schedule_date]
            day_name = date.fromisoformat(schedule_date).strftime("%A")
            day_hours = day_df["Hours"].sum()
            done_today = int(day_df["Completed"].sum())
            with st.expander(
                f"{day_name}, {date.fromisoformat(schedule_date).strftime('%d %b %Y')} "
                f"· {day_hours:.1f} hours · {done_today}/{len(day_df)} completed",
                expanded=(schedule_date == visible_dates[0]),
            ):
                for _, row in day_df.iterrows():
                    task_id = row["task_id"]
                    checkbox_col, details_col = st.columns([0.08, 0.92])
                    with checkbox_col:
                        is_done = st.checkbox(
                            "Completed",
                            value=task_id in st.session_state.completed_tasks,
                            key=f"done_{task_id}",
                            label_visibility="collapsed",
                        )
                    if is_done and task_id not in st.session_state.completed_tasks:
                        st.session_state.completed_tasks.append(task_id)
                    elif not is_done and task_id in st.session_state.completed_tasks:
                        st.session_state.completed_tasks.remove(task_id)

                    with details_col:
                        status_icon = "✓" if is_done else "○"
                        st.markdown(
                            f"**{status_icon} {row['Subject']} — {row['Topic / Task']}**  \n"
                            f'<span class="pill">{row["Type"]}</span> '
                            f'<span class="muted">{row["Hours"]} hours</span>',
                            unsafe_allow_html=True,
                        )
                        if row["Notes"]:
                            st.caption(row["Notes"])
                    st.divider()

    # Refresh progress after checkbox changes in the current render.
    refreshed_df = task_dataframe()
    refreshed_completed = int(refreshed_df["Completed"].sum()) if not refreshed_df.empty else 0
    refreshed_progress = refreshed_completed / total_tasks if total_tasks else 0
    st.progress(
        refreshed_progress,
        text=f"Live progress: {refreshed_completed} of {total_tasks} tasks completed",
    )

    st.markdown("### 04 / Export your timetable")
    export_df = task_dataframe()
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "⬇ Download schedule as CSV",
            data=export_csv(export_df),
            file_name="studyflow_study_plan.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with export_right:
        st.download_button(
            "⬇ Download schedule as TXT",
            data=export_txt(export_df),
            file_name="studyflow_study_plan.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    if st.button("↻ Create a new study plan", use_container_width=True):
        st.session_state.plan_data = None
        st.session_state.plan_inputs = None
        st.session_state.completed_tasks = []
        st.session_state.plan_created_at = None
        st.rerun()

    st.caption(
        "Your checklist and generated plan are stored in Streamlit session state for this active session. "
        "They are not a permanent account or cross-device storage system."
    )


st.markdown(
    '<div style="text-align:center;color:#9299aa;font-size:.78rem;padding:2rem 0 1rem">STUDYFLOW AI · MAKE EVERY STUDY SESSION COUNT</div>',
    unsafe_allow_html=True,
)
