# -------------------------------
# Employee Task Dashboard + Gemini Ops Assistant (multi-turn chat + tool calls)
# -------------------------------
# pip install -U streamlit pandas plotly google-auth gspread google-generativeai python-dateutil
# export GOOGLE_API_KEY="your_gemini_api_key"

import os
import re
import json
import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from dateutil import parser as dtparser

# ========= EDIT THESE CONSTANTS =========
JSON_PATH = "fes-employee-eda-01c012142a64.json"  # your service-account JSON path
SHEET_URL = "https://docs.google.com/spreadsheets/d/1yaW7V7hSBqOBZYbqIUKsGrhB8pVtWWrzq7t5scq3JVI/edit?gid=0#gid=0"
SHEET_GID = "0"        # optional; if you want a specific tab by gid
SHEET_TITLE = None     # optional; if set, takes priority over gid
# =======================================

# ---- Google scopes: read+write (agent can update status/notes) ----
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.set_page_config(layout="wide")
st.title("📋 Employee Task Dashboard (Interactive with Plotly + 🤖 Gemini Assistant)")

# ---------------- Helpers ----------------
def parse_date(x):
    """
    Robust date parser for CSV/Google Sheet cells.
    Handles ordinal strings, day-first, ISO, Excel/Sheets serials, missing year -> current year.
    Returns pd.Timestamp or pd.NaT.
    """
    # empties
    if x is None:
        return pd.NaT
    if isinstance(x, float) and pd.isna(x):
        return pd.NaT
    if isinstance(x, str) and x.strip() == "":
        return pd.NaT

    # already datetime-like
    if isinstance(x, (pd.Timestamp, datetime.datetime)):
        ts = pd.to_datetime(x, errors="coerce")
        return ts.tz_localize(None) if isinstance(ts, pd.Timestamp) and ts.tz is not None else ts
    if isinstance(x, datetime.date):
        return pd.Timestamp(x)

    # Excel/Sheets serial
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            val = float(x)
            if 20000 <= val <= 60000:  # ~1954..2064
                base = datetime.datetime(1899, 12, 30)
                return pd.to_datetime(base + datetime.timedelta(days=val))
        except Exception:
            pass

    # strings
    s = str(x).strip()
    if s.lower() in {"na", "n/a", "none", "null", "-", "—", "tbd", "not set"}:
        return pd.NaT
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", s, flags=re.IGNORECASE)  # remove ordinals
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(\d{1,2})\s*[-–—]\s*\d{1,2}(\b| )", r"\1 ", s)       # "1-3 July" -> "1 July"

    default_dt = datetime.datetime(datetime.date.today().year, 1, 1)
    for try_fn in (
        lambda v: pd.to_datetime(v, dayfirst=True, errors="raise"),
        lambda v: pd.to_datetime(v, errors="raise"),
        lambda v: pd.to_datetime(dtparser.parse(v, dayfirst=True, fuzzy=True, default=default_dt)),
    ):
        try:
            ts = try_fn(s)
            if isinstance(ts, pd.Timestamp):
                return ts.tz_localize(None) if ts.tz is not None else ts
            return pd.to_datetime(ts)
        except Exception:
            continue
    return pd.NaT

def urgency_score(days):
    if pd.isna(days): return 0
    try: d = int(days)
    except Exception: return 0
    if d <= 0: return 100
    if d <= 1: return 90
    if d <= 3: return 80
    if d <= 7: return 60
    if d <= 14: return 40
    if d <= 30: return 20
    return 0

def gs_client():
    creds = Credentials.from_service_account_file(JSON_PATH, scopes=SCOPES)
    return gspread.authorize(creds)

def open_sheet():
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", SHEET_URL)
    if not m: raise ValueError("Invalid Google Sheet URL")
    return gs_client().open_by_key(m.group(1))

@st.cache_data(ttl=300)
def load_sheet_from_gdrive(sheet_gid: str | None = None, sheet_title: str | None = None) -> pd.DataFrame:
    """Read the Google Sheet into a DataFrame."""
    sh = open_sheet()
    ws = None
    if sheet_title:
        ws = sh.worksheet(sheet_title)
    elif sheet_gid is not None:
        for w in sh.worksheets():
            if str(w.id) == str(sheet_gid):
                ws = w
                break
    if ws is None:
        ws = sh.sheet1

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values[1:], columns=values[0])
    return df

# ---------- Data source: CSV upload OR constant Google Sheet ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    with st.spinner("Loading live Google Sheet…"):
        df = load_sheet_from_gdrive(sheet_gid=SHEET_GID, sheet_title=SHEET_TITLE)

if df is None or df.empty:
    st.info("No data found. Upload a CSV or check your Google Sheet link/permissions.")
    st.stop()

# ---------- Clean & prepare ----------
df.columns = df.columns.str.strip()

# Drop unnecessary columns
drop_cols = [col for col in df.columns if "Unnamed" in col or "Design Drive Link" in col]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Remove month-label rows in Content Type (e.g., "April 2025")
if "Content Type" in df.columns:
    df = df[df["Content Type"].notna()]
    df = df[~df["Content Type"].astype(str).str.contains(r"20[0-9]{2}", na=False)]

# Dates
if "Assigned Date" in df.columns:
    df["Assigned Date"] = df["Assigned Date"].apply(parse_date)
if "Completion Date" in df.columns:
    df["Completion Date"] = df["Completion Date"].apply(parse_date)
if "Deadline" in df.columns:
    df["Deadline"] = df["Deadline"].apply(parse_date)

# Status
if "Design Status" in df.columns:
    df["Design Status"] = df["Design Status"].fillna("Non Completed").astype(str).str.strip()
else:
    df["Design Status"] = df.get("Status", "Non Completed").fillna("Non Completed").astype(str).str.strip()

df["Status"] = df["Design Status"].apply(lambda x: "Completed" if str(x).strip().lower()=="completed" else "Remaining")

# Normalize content type
if "Content Type" in df.columns:
    df["Content Type (norm)"] = (
        df["Content Type"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["Is Event Cover"] = df["Content Type (norm)"].str.contains(r"\bevent cover\b", na=False)
else:
    df["Content Type (norm)"] = ""
    df["Is Event Cover"] = False

# ---------- Sidebar filters ----------
designers = ["All"] + (sorted(df["Designer Name"].dropna().unique()) if "Designer Name" in df.columns else [])
assigners = ["All"] + (sorted(df["Assigned By"].dropna().unique()) if "Assigned By" in df.columns else [])
selected_designer = st.sidebar.selectbox("🎨 Filter by Designer", designers, index=0 if designers else None)
selected_by = st.sidebar.selectbox("🧑‍💼 Filter by Assigned By", assigners, index=0 if assigners else None)

min_date = df["Assigned Date"].min() if "Assigned Date" in df.columns else pd.NaT
max_date = df["Assigned Date"].max() if "Assigned Date" in df.columns else pd.NaT

today = datetime.date.today()
start_default = (min_date.date() if pd.notna(min_date) else today)
end_default = (max_date.date() if pd.notna(max_date) else today)
date_selection = st.sidebar.date_input("📅 Date Range", value=(start_default, end_default))

# Apply filters
filtered = df.copy()
if "Designer Name" in filtered.columns and selected_designer and selected_designer != "All":
    filtered = filtered[filtered["Designer Name"] == selected_designer]
if "Assigned By" in filtered.columns and selected_by and selected_by != "All":
    filtered = filtered[filtered["Assigned By"] == selected_by]

# Date range filter
start_date, end_date = (None, None)
if isinstance(date_selection, tuple) and len(date_selection) == 2:
    start_date, end_date = date_selection
elif hasattr(date_selection, "year"):
    start_date = end_date = date_selection

if start_date and end_date and "Assigned Date" in filtered.columns:
    filtered = filtered[
        (filtered["Assigned Date"] >= pd.to_datetime(start_date)) &
        (filtered["Assigned Date"] <= pd.to_datetime(end_date))
    ]

# ---------- KPIs ----------
total_tasks = len(filtered)
completed_mask = (filtered["Status"] == "Completed")
completed_count = int(completed_mask.sum())
remaining_count = int(total_tasks - completed_count)
completion_pct = round((completed_count / total_tasks * 100), 1) if total_tasks else 0

col1, col2, col3 = st.columns(3)
col1.metric("📌 Total Tasks", total_tasks)
col2.metric("✅ Completed", completed_count)
col3.metric("❌ Remaining", remaining_count)

st.markdown(f"### 🔄 Completion Progress: **{completion_pct}%**")
st.progress(completion_pct / 100 if total_tasks else 0)

# =========================
# 🔥 Priority Tasks (Deadline + Event Cover, with smart fallback)
# =========================
st.subheader("🔥 Priority Tasks")

prio = filtered.copy()
if "Status" in prio.columns:
    prio = prio[prio["Status"] != "Completed"]

# Days to Deadline
if "Deadline" in prio.columns and not pd.api.types.is_datetime64_any_dtype(prio["Deadline"]):
    prio["Deadline"] = prio["Deadline"].apply(parse_date)
if "Deadline" in prio.columns:
    today_midnight = pd.Timestamp(datetime.date.today())
    prio["Days to Deadline"] = (prio["Deadline"] - today_midnight).dt.days
else:
    prio["Days to Deadline"] = pd.NA
has_any_deadline = "Days to Deadline" in prio.columns and prio["Days to Deadline"].notna().any()

# Event Cover flag (already normalized on df)
if "Is Event Cover" not in prio.columns and "Content Type" in prio.columns:
    ct_norm = prio["Content Type"].astype(str).str.lower().str.replace(r"\s+"," ", regex=True).str.strip()
    prio["Is Event Cover"] = ct_norm.str.contains(r"\bevent cover\b", na=False)

# Scoring controls
ctrl1, ctrl2 = st.columns([1, 1])
with ctrl1:
    top_n_prio = st.slider("Show Top N priority tasks", min_value=5, max_value=50, value=10, step=1)
with ctrl2:
    event_boost = st.slider("Event Cover boost", 0, 100, 50, 5,
                            help="Extra points for tasks whose Content Type is Event Cover.")

if has_any_deadline:
    prio["Urgency Score"] = prio["Days to Deadline"].apply(urgency_score)
    prio["Priority Score"] = prio["Urgency Score"] + prio["Is Event Cover"].astype(int) * event_boost
    sort_cols = ["Priority Score", "Days to Deadline"]
    sort_asc = [False, True]
    if "Assigned Date" in prio.columns:
        sort_cols.append("Assigned Date")
        sort_asc.append(True)
    prio = prio.sort_values(by=sort_cols, ascending=sort_asc, na_position="last")
else:
    prio["Priority Score"] = prio["Is Event Cover"].astype(int) * max(event_boost, 1)
    if "Assigned Date" in prio.columns:
        prio = prio.sort_values(by=["Priority Score", "Assigned Date"], ascending=[False, True], na_position="last")
    else:
        prio = prio.sort_values(by=["Priority Score"], ascending=[False], na_position="last")
    st.caption("No deadlines found — prioritizing Event Cover tasks first, then oldest assigned.")

if "Content Title" in prio.columns:
    prio["Title (short)"] = prio["Content Title"].astype(str).str.slice(0, 60).where(
        prio["Content Title"].astype(str).str.len() <= 60,
        prio["Content Title"].astype(str).str.slice(0, 57) + "…"
    )

display_cols = [c for c in [
    "Title (short)" if "Title (short)" in prio.columns else None,
    "Content Title" if "Content Title" in prio.columns else None,
    "Designer Name" if "Designer Name" in prio.columns else None,
    "Assigned By" if "Assigned By" in prio.columns else None,
    "Content Type" if "Content Type" in prio.columns else None,
    "Assigned Date" if "Assigned Date" in prio.columns else None,
    "Deadline" if "Deadline" in prio.columns else None,
    "Days to Deadline" if "Days to Deadline" in prio.columns else None,
    "Is Event Cover",
    "Priority Score"
] if c]

c_tbl, c_bar = st.columns([1.3, 1])
with c_tbl:
    if prio.empty:
        st.info("No remaining tasks to prioritize for the current filters.")
    else:
        st.dataframe(prio.head(top_n_prio)[display_cols], use_container_width=True, hide_index=True)

with c_bar:
    show_chart = prio.head(top_n_prio).copy()
    y_field = "Title (short)" if "Title (short)" in show_chart.columns else (
        "Content Title" if "Content Title" in show_chart.columns else None
    )
    if y_field is not None:
        fig_prio = px.bar(
            show_chart, y=y_field, x="Priority Score", orientation="h",
            color="Is Event Cover",
            color_discrete_map={True: "#d62728", False: "#1f77b4"},
            text="Priority Score",
            hover_data=[c for c in ["Deadline", "Days to Deadline", "Content Type"] if c in show_chart.columns],
            height=max(350, 26 * len(show_chart))
        )
        fig_prio.update_layout(
            xaxis_title="Priority", yaxis_title="", legend_title="Event Cover",
            margin=dict(l=10, r=10, t=10, b=10), bargap=0.25
        )
        fig_prio.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_prio, use_container_width=True)
    else:
        st.caption("Add a 'Content Title' column for a nicer priority chart.")

# 📊 Task Completion Overview
st.subheader("📊 Task Completion Overview")
status_counts = filtered["Status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]
fig1 = px.bar(
    status_counts, x="Status", y="Count", color="Status", text="Count",
    color_discrete_map={"Completed": "green", "Remaining": "red"}
)
fig1.update_layout(xaxis_title="", yaxis_title="Tasks", height=400)
st.plotly_chart(fig1, use_container_width=True)

# 🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining
st.subheader("🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining")
compare_opts = [c for c in ["Designer Name", "Assigned By"] if c in filtered.columns]
if compare_opts:
    compare_dim = st.selectbox("Compare by", options=compare_opts, index=0)
    filtered_comp = filtered.copy()
    filtered_comp[compare_dim] = filtered_comp[compare_dim].fillna("Unknown")

    team_counts = (
        filtered_comp.groupby([compare_dim, "Status"]).size().reset_index(name="Count")
    )
    all_editors = team_counts[compare_dim].unique()
    status_order = ["Completed", "Remaining"]
    full_index = pd.MultiIndex.from_product([all_editors, status_order], names=[compare_dim, "Status"])
    team_counts = (
        team_counts.set_index([compare_dim, "Status"]).reindex(full_index, fill_value=0).reset_index()
    )
    pivot_for_sort = team_counts.pivot(index=compare_dim, columns="Status", values="Count").fillna(0)
    sorted_editors = pivot_for_sort.sort_values(by="Completed", ascending=False).index.tolist()
    n_editors = len(sorted_editors)

    if n_editors == 0:
        st.info("No editors to compare for the current filters.")
    else:
        if n_editors == 1:
            st.caption("Only one editor in the current filter. Showing all results.")
            top_n = 1
        else:
            top_n = st.slider("Show Top N", 1, n_editors, min(10, n_editors))
        keep_editors = set(sorted_editors[:top_n])
        team_counts_top = team_counts[team_counts[compare_dim].isin(keep_editors)]

        fig_team = px.bar(
            team_counts_top, y=compare_dim, x="Count", color="Status", orientation="h",
            category_orders={compare_dim: [e for e in sorted_editors if e in keep_editors], "Status": status_order},
            text="Count", barmode="group",
            color_discrete_map={"Completed": "green", "Remaining": "red"},
            height=max(400, 30 * len(keep_editors))
        )
        fig_team.update_layout(
            xaxis_title="Tasks", yaxis_title="", legend_title="", bargap=0.25,
            margin=dict(l=120, r=20, t=40, b=20),
        )
        fig_team.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_team, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**🏆 Top Completed**")
            top_completed = (
                pivot_for_sort["Completed"].sort_values(ascending=False).head(top_n)
                .reset_index().rename(columns={compare_dim: "Editor", "Completed": "Completed Tasks"})
            )
            st.dataframe(top_completed, use_container_width=True, hide_index=True)

        with colB:
            st.markdown("**⏳ Most Remaining**")
            top_remaining = (
                pivot_for_sort["Remaining"].sort_values(ascending=False).head(top_n)
                .reset_index().rename(columns={compare_dim: "Editor", "Remaining": "Remaining Tasks"})
            )
            st.dataframe(top_remaining, use_container_width=True, hide_index=True)

# 📆 Assignment vs Completion Timeline
st.subheader("📆 Assignment vs Completion Timeline")
tmp = filtered.copy()
tmp["Assigned Day"] = tmp["Assigned Date"].dt.date if "Assigned Date" in tmp.columns else pd.NaT
tmp["Completed Day"] = tmp["Completion Date"].dt.date if "Completion Date" in tmp.columns else pd.NaT
assigned = tmp.groupby("Assigned Day").size().reset_index(name="Assigned")
completed = tmp.groupby("Completed Day").size().reset_index(name="Completed")
timeline = pd.merge(assigned, completed, left_on="Assigned Day", right_on="Completed Day", how="outer")
timeline["Day"] = timeline["Assigned Day"].combine_first(timeline["Completed Day"])
timeline = timeline.fillna(0).sort_values("Day")
timeline = timeline[["Day", "Assigned", "Completed"]]
fig2 = px.line(timeline, x="Day", y=["Assigned", "Completed"], markers=True)
fig2.update_layout(yaxis_title="Tasks", height=450)
st.plotly_chart(fig2, use_container_width=True)

# 📂 Content Type Breakdown
if "Content Type" in filtered.columns:
    st.subheader("📂 Tasks by Content Type")
    content_counts = filtered["Content Type"].value_counts().reset_index()
    content_counts.columns = ["Content Type", "Count"]
    fig3 = px.bar(content_counts, y="Content Type", x="Count", orientation="h", text="Count", height=600)
    fig3.update_layout(yaxis_title="", xaxis_title="Tasks", margin=dict(l=100))
    st.plotly_chart(fig3, use_container_width=True)

# 📅 Weekday Assignment Distribution
if "Assigned Date" in filtered.columns:
    st.subheader("📅 Tasks by Weekday")
    filtered["Weekday"] = filtered["Assigned Date"].dt.day_name()
    weekday_counts = (
        filtered["Weekday"].value_counts()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .fillna(0).reset_index()
    )
    weekday_counts.columns = ["Weekday", "Count"]
    fig4 = px.bar(weekday_counts, x="Weekday", y="Count", text="Count", color="Weekday", height=400)
    st.plotly_chart(fig4, use_container_width=True)

# 📋 Task Tables
table_cols = [c for c in ["Content Title", "Designer Name", "Assigned By", "Assigned Date",
                          "Deadline", "Completion Date", "Design Status"] if c in filtered.columns]
if table_cols:
    st.subheader("📋 Completed Tasks")
    st.dataframe(filtered[filtered["Status"] == "Completed"][table_cols], use_container_width=True)
    st.subheader("📋 Remaining Tasks")
    st.dataframe(filtered[filtered["Status"] == "Remaining"][table_cols], use_container_width=True)

# ===========================================================
# 🤖 Gemini 2.0 Flash Assistant (tools + routing, multi-turn chat)
# ===========================================================
st.markdown("---")
st.subheader("🤖 Task Assistant (Gemini 2.0 Flash)")

# ---- Tool implementations (read/write the sheet) ----
def _read_df(worksheet_title=None):
    sh = open_sheet()
    ws = sh.worksheet(worksheet_title) if worksheet_title else sh.sheet1
    values = ws.get_all_values()
    if not values:
        empty = pd.DataFrame()
        empty["__row"] = []
        return empty, ws
    df_local = pd.DataFrame(values[1:], columns=values[0])
    df_local.columns = df_local.columns.str.strip()
    for c in ["Assigned Date", "Completion Date", "Deadline"]:
        if c in df_local.columns:
            df_local[c] = df_local[c].apply(parse_date)
    if "Design Status" in df_local.columns:
        s = df_local["Design Status"].fillna("").astype(str).str.strip().str.lower()
        df_local["Status"] = s.apply(lambda x: "Completed" if x == "completed" else "Remaining")
    else:
        df_local["Status"] = "Remaining"
    if "Content Type" in df_local.columns:
        norm = df_local["Content Type"].astype(str).str.lower().str.replace(r"\s+"," ", regex=True).str.strip()
        df_local["Is Event Cover"] = norm.str.contains(r"\bevent cover\b", na=False)
    else:
        df_local["Is Event Cover"] = False
    df_local["__row"] = df_local.index + 2
    return df_local, ws

def tool_list_due(days: int = 3) -> str:
    dfl, _ = _read_df()
    if "Deadline" not in dfl.columns:
        return "No 'Deadline' column found."
    today = pd.Timestamp(datetime.date.today()).floor("D")
    dfl["Days to Deadline"] = (dfl["Deadline"] - today).dt.days
    R = dfl[dfl["Status"] != "Completed"]
    overdue = R[R["Days to Deadline"] < 0]
    today_due = R[R["Days to Deadline"] == 0]
    soon = R[R["Days to Deadline"].between(1, days, inclusive="both")]
    def _fmt(sub, label):
        if sub.empty: return f"- {label}: none"
        titles = (sub["Content Title"] if "Content Title" in sub.columns else sub.index.astype(str)).head(10)
        return f"- {label}: {len(sub)}. Top: " + "; ".join(map(str, titles))
    return "\n".join([_fmt(overdue, "Overdue"), _fmt(today_due, "Due today"), _fmt(soon, f"Due in next {days} days")])

def tool_list_priority(top_n: int = 10, event_boost: int = 50) -> list:
    dfl, _ = _read_df()
    if "Deadline" in dfl.columns:
        today = pd.Timestamp(datetime.date.today()).floor("D")
        dfl["Days to Deadline"] = (dfl["Deadline"] - today).dt.days
    else:
        dfl["Days to Deadline"] = pd.NA
    dfl["Urgency"] = dfl["Days to Deadline"].apply(urgency_score)
    dfl["Priority"] = dfl["Urgency"] + dfl["Is Event Cover"].astype(int) * int(event_boost)
    R = dfl[dfl["Status"] != "Completed"].copy()
    sort_cols = ["Priority", "Days to Deadline"]
    sort_asc = [False, True]
    if "Assigned Date" in R.columns:
        sort_cols.append("Assigned Date"); sort_asc.append(True)
    R = R.sort_values(by=sort_cols, ascending=sort_asc, na_position="last")
    keep = [c for c in ["Task ID", "Content Title", "Designer Name", "Deadline", "Days to Deadline",
                         "Is Event Cover", "Priority", "Status"] if c in R.columns]
    return R.head(int(top_n))[keep].to_dict(orient="records")

def tool_update_status(task_id: str, new_status: str, confirm: bool = False) -> str:
    if not confirm:
        return "Dry-run: set confirm=True to execute."
    dfl, ws = _read_df()
    if "Task ID" not in dfl.columns:
        return "No 'Task ID' column in sheet."
    hits = dfl.index[dfl["Task ID"].astype(str) == str(task_id)].tolist()
    if not hits:
        return f"Task ID {task_id} not found."
    row = int(dfl.loc[hits[0], "__row"])
    headers = ws.row_values(1)
    if "Design Status" not in headers:
        return "Column 'Design Status' not found."
    col = headers.index("Design Status") + 1
    norm = "Completed" if str(new_status).strip().lower() == "completed" else "Remaining"
    ws.update_cell(row, col, norm)
    return f"Updated Task ID {task_id} → {norm} (row {row})."

def tool_add_note(task_id: str, note: str, confirm: bool = False) -> str:
    if not confirm:
        return "Dry-run: set confirm=True to execute."
    dfl, ws = _read_df()
    if "Task ID" not in dfl.columns:
        return "No 'Task ID' column in sheet."
    hits = dfl.index[dfl["Task ID"].astype(str) == str(task_id)].tolist()
    if not hits:
        return f"Task ID {task_id} not found."
    row = int(dfl.loc[hits[0], "__row"])
    headers = ws.row_values(1)
    if "Notes" not in headers:
        ws.update_cell(1, len(headers) + 1, "Notes")
        headers.append("Notes")
    col = headers.index("Notes") + 1
    current = ws.cell(row, col).value or ""
    stamp = f"[{datetime.date.today()}] "
    ws.update_cell(row, col, (current + " | " if current else "") + stamp + note)
    return f"Added note to Task ID {task_id}."

# ---- Gemini tool schemas (UPPERCASE types, no min/max/default) ----
TOOLS_SPEC = [
    {
      "name": "tool_list_due",
      "description": "Summarize overdue, due-today, and due-soon tasks (Remaining only).",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "days": { "type": "INTEGER", "description": "How many days ahead to consider 'due soon' (e.g., 3)." }
        },
        "required": []
      }
    },
    {
      "name": "tool_list_priority",
      "description": "Return top priority tasks as list of objects (deadline urgency + Event Cover boost).",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "top_n":      { "type": "INTEGER", "description": "How many tasks to return (e.g., 10)." },
          "event_boost":{ "type": "INTEGER", "description": "Extra points for Event Cover tasks (e.g., 50)." }
        },
        "required": []
      }
    },
    {
      "name": "tool_update_status",
      "description": "Update Design Status for a Task ID to Completed/Remaining. confirm=True required to write.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "task_id":    { "type": "STRING",  "description": "The Task ID to update." },
          "new_status": { "type": "STRING",  "enum": ["Completed","Remaining","completed","remaining"] },
          "confirm":    { "type": "BOOLEAN", "description": "Must be true to perform the write." }
        },
        "required": ["task_id","new_status"]
      }
    },
    {
      "name": "tool_add_note",
      "description": "Append a note to a task's 'Notes' column. confirm=True required to write.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "task_id": { "type": "STRING",  "description": "The Task ID to update." },
          "note":    { "type": "STRING",  "description": "The note text to append." },
          "confirm": { "type": "BOOLEAN", "description": "Must be true to perform the write." }
        },
        "required": ["task_id","note"]
      }
    },
]

# ---- Gemini model creation (AUTO mode + fallback) ----
def _make_model():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Set your Gemini API key in the environment.")
    genai.configure(api_key=api_key)

    names = [t["name"] for t in TOOLS_SPEC]
    sys_inst = (
        "You are a helpful ops assistant for a design team. "
        "Only call tools when the user clearly asks for data from the sheet or to make a change. "
        "For casual conversation, respond without calling tools. "
        "When you call a tool, ALWAYS supply every required parameter. "
        "If you don't have the required parameters, ask a concise clarifying question instead of calling any tool. "
        "Only write to the sheet when confirm=True. Never guess Task IDs. Prefer concise answers."
    )

    # Try NEW schema first
    try:
        return genai.GenerativeModel(
            "gemini-2.0-flash",
            tools=TOOLS_SPEC,
            tool_config={ "function_calling_config": { "mode": "AUTO", "allowed_function_names": names } },
            system_instruction=sys_inst,
        )
    except Exception:
        # Fallback for older SDKs
        return genai.GenerativeModel(
            "gemini-2.0-flash",
            tools=TOOLS_SPEC,
            tool_config={ "allowed_function_names": names },
            system_instruction=sys_inst,
        )

# Keep a single chat in session for multi‑turn
if "gem_model" not in st.session_state:
    try:
        st.session_state.gem_model = _make_model()
        st.session_state.gem_chat = st.session_state.gem_model.start_chat()
    except Exception as e:
        st.warning(f"Gemini init: {e}")
        st.session_state.gem_model = None
        st.session_state.gem_chat = None

# Normalize tool‑call args across SDKs
def _extract_args(fc):
    # fc.args may be dict or proto list
    try:
        if isinstance(fc.args, dict):
            return fc.args
        args = {}
        for kv in fc.args:
            val = kv.value
            try:
                val = json.loads(val) if isinstance(val, str) else val
            except Exception:
                pass
            args[kv.key] = val
        return args
    except Exception:
        return {}

# Send tool result back to the model in the correct format (SDK-compatible)
def _send_tool_result(chat, name: str, result):
    """
    Sends a function result to Gemini using the proper 'function_response' shape.
    Works across SDK versions.
    """
    # Ensure response is JSON-serializable
    if isinstance(result, (str, int, float)):
        resp_obj = {"result": result}
    else:
        try:
            json.dumps(result, default=str)
            resp_obj = result
        except Exception:
            resp_obj = {"result": str(result)}

    tool_msg = {
        "function_response": {
            "name": name,
            "response": resp_obj
        }
    }
    try:
        return chat.send_message(tool_msg)
    except Exception:
        try:
            from google.generativeai.types import FunctionResponse
            return chat.send_message(FunctionResponse(name=name, response=resp_obj))
        except Exception:
            return chat.send_message(f"{name} -> {json.dumps(resp_obj, default=str)}")

def _iter_parts(resp):
    """Yield all parts from the first candidate safely."""
    try:
        for p in getattr(resp.candidates[0].content, "parts", []) or []:
            yield p
    except Exception:
        return

def _get_first_function_call(resp):
    """Return the first function_call part if present, else None."""
    for p in _iter_parts(resp):
        if hasattr(p, "function_call") and p.function_call:
            return p.function_call
    return None

def _get_text_safe(resp) -> str:
    """Return assistant text even if resp.text raises due to function_call-only parts."""
    # Try the SDK's convenience
    try:
        t = getattr(resp, "text", None)
        if t:
            return t
    except Exception:
        pass

    # Manual: concatenate any text parts
    texts = []
    for p in _iter_parts(resp):
        # Newer SDKs: text part as attribute
        if getattr(p, "text", None):
            texts.append(p.text)
            continue
        # Older SDKs: dict-like or inline data
        try:
            if isinstance(p, dict) and "text" in p:
                texts.append(str(p["text"]))
        except Exception:
            pass
    return "\n".join(texts).strip() or "(no response)"


# Validate & dispatch tool calls safely (prevents missing-args crashes)
def _dispatch_tool(name: str, args: dict):
    args = args or {}

    if name == "tool_list_due":
        days = args.get("days", 3)
        try:
            days = int(days)
        except Exception:
            days = 3
        return True, tool_list_due(days=days)

    if name == "tool_list_priority":
        top_n = args.get("top_n", 10)
        event_boost = args.get("event_boost", 50)
        try:
            top_n = int(top_n)
        except Exception:
            top_n = 10
        try:
            event_boost = int(event_boost)
        except Exception:
            event_boost = 50
        return True, tool_list_priority(top_n=top_n, event_boost=event_boost)

    if name == "tool_update_status":
        task_id = args.get("task_id")
        new_status = args.get("new_status")
        confirm = bool(args.get("confirm", False))
        if not task_id or not new_status:
            return False, {
                "error": "missing_args",
                "required": ["task_id", "new_status"],
                "hint": "Call tool_update_status with task_id and new_status. "
                        "Example: {'task_id':'123','new_status':'Completed','confirm':true}"
            }
        return True, tool_update_status(task_id=str(task_id), new_status=str(new_status), confirm=confirm)

    if name == "tool_add_note":
        task_id = args.get("task_id")
        note = args.get("note")
        confirm = bool(args.get("confirm", False))
        if not task_id or note in (None, ""):
            return False, {
                "error": "missing_args",
                "required": ["task_id", "note"],
                "hint": "Call tool_add_note with task_id and note. "
                        "Example: {'task_id':'123','note':'brief updated','confirm':true}"
            }
        return True, tool_add_note(task_id=str(task_id), note=str(note), confirm=confirm)

    return False, {"error": "unknown_tool", "name": name}

# Route tool calls and send tool results back to the model until we get final text
def _chat_send_and_handle_tools(user_text: str) -> str:
    chat = st.session_state.gem_chat
    resp = chat.send_message(user_text)

    # Loop while model keeps asking for tools
    safety_hops = 0
    while getattr(resp, "candidates", None) and safety_hops < 8:
        fc = _get_first_function_call(resp)
        if not fc:
            break  # no tool calls -> we should have final text

        name = fc.name
        args = _extract_args(fc)

        # Validate + dispatch
        ok, result = _dispatch_tool(name, args)

        # Send function result back to the model (correct envelope)
        resp = _send_tool_result(chat, name, result)

        safety_hops += 1

    # Safely return text (even if SDK can't render resp.text)
    return _get_text_safe(resp)

# ---- Chat UI ----
st.caption(f"google-generativeai version: {getattr(genai, '__version__', 'unknown')}")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me what's due, show priority tasks, or say: update task 123 to Completed confirm=True"}]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_q = st.chat_input("Type here… (e.g., 'what's overdue today', 'top 5 priority', 'update task 123 to Completed confirm=True')")
if user_q and st.session_state.gem_chat:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                out = _chat_send_and_handle_tools(user_q.strip())
            except Exception as e:
                out = f"Error: {e}"
            st.markdown(out)
            st.session_state.messages.append({"role": "assistant", "content": out})

elif user_q and not st.session_state.gem_chat:
    st.warning("Gemini assistant didn’t initialize (check GOOGLE_API_KEY). Dashboard above still works.")
