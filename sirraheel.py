import streamlit as st
import pandas as pd
import plotly.express as px
from dateutil import parser
import datetime
import re
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")
st.title("📋 Employee Task Dashboard (Interactive with Plotly)")

# Helper to parse dates
def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    try:
        return pd.to_datetime(parser.parse(str(date_str), dayfirst=True, fuzzy=True))
    except Exception:
        return None

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Drop unnecessary columns
    drop_cols = [col for col in df.columns if "Unnamed" in col or "Design Drive Link" in col]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Remove month label rows like "April 2025"
    if "Content Type" in df.columns:
        df = df[df["Content Type"].notna()]
        df = df[~df["Content Type"].astype(str).str.contains(r"20[0-9]{2}", na=False)]

    # Parse dates and clean status
    df["Assigned Date"] = df["Assigned Date"].apply(parse_date)
    df["Completion Date"] = df["Completion Date"].apply(parse_date)
    df["Design Status"] = df["Design Status"].fillna("Non Completed").astype(str).str.strip()

    # Only mark as Completed if exactly "completed"
    df["Status"] = df["Design Status"].apply(
        lambda x: "Completed" if str(x).strip().lower() == "completed" else "Remaining"
    )

    # Sidebar filters
    designers = ["All"] + sorted(df["Designer Name"].dropna().unique())
    assigners = ["All"] + sorted(df["Assigned By"].dropna().unique())
    selected_designer = st.sidebar.selectbox("🎨 Filter by Designer", designers)
    selected_by = st.sidebar.selectbox("🧑‍💼 Filter by Assigned By", assigners)

    min_date = df["Assigned Date"].min()
    max_date = df["Assigned Date"].max()

    # Always pass a valid 2-tuple to date_input; handle NaT by falling back to today
    today = datetime.date.today()
    start_default = (min_date.date() if pd.notna(min_date) else today)
    end_default = (max_date.date() if pd.notna(max_date) else today)

    date_selection = st.sidebar.date_input("📅 Date Range", value=(start_default, end_default))

    # Apply filters
    filtered = df.copy()
    if selected_designer != "All":
        filtered = filtered[filtered["Designer Name"] == selected_designer]
    if selected_by != "All":
        filtered = filtered[filtered["Assigned By"] == selected_by]

    # Date range filter (supports single-date or range)
    start_date, end_date = (None, None)
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start_date, end_date = date_selection
    elif hasattr(date_selection, "year"):  # single date picked
        start_date = end_date = date_selection

    if start_date and end_date:
        filtered = filtered[
            (filtered["Assigned Date"] >= pd.to_datetime(start_date)) &
            (filtered["Assigned Date"] <= pd.to_datetime(end_date))
        ]

    # Task counts
    total_tasks = len(filtered)
    completed_mask = filtered["Status"] == "Completed"
    completed_count = int(completed_mask.sum())
    remaining_count = int(total_tasks - completed_count)
    completion_pct = round((completed_count / total_tasks * 100), 1) if total_tasks else 0

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("📌 Total Tasks", total_tasks)
    col2.metric("✅ Completed", completed_count)
    col3.metric("❌ Remaining", remaining_count)

    st.markdown(f"### 🔄 Completion Progress: **{completion_pct}%**")
    st.progress(completion_pct / 100 if total_tasks else 0)
    


    # =========================
# 🔥 Priority Tasks (Deadline + Event Cover)
# =========================
st.subheader("🔥 Priority Tasks")

# 1) Let user choose the deadline column (auto-detect common names)
deadline_candidates_all = ["Deadline", "Due Date", "Target Date", "Event Date", "Publish Date", "Publishing Date"]
deadline_candidates = [c for c in deadline_candidates_all if c in df.columns]

deadline_col = None
if deadline_candidates:
    deadline_col = st.sidebar.selectbox(
        "🗓️ Deadline column",
        options=deadline_candidates,
        index=0,
        help="Pick the column that holds task deadlines."
    )

# 2) Work on remaining tasks only
prio = filtered.copy()
if "Status" in prio.columns:
    prio = prio[prio["Status"] != "Completed"]

# 3) Parse deadline and compute days to deadline
today_date = datetime.date.today()
if deadline_col:
    # parse if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(prio[deadline_col]):
        prio[deadline_col] = prio[deadline_col].apply(parse_date)
    prio["Days to Deadline"] = (prio[deadline_col].dt.date - today_date).dt.days
else:
    prio["Days to Deadline"] = pd.NA  # no deadline available

# 4) Flag Event Cover
if "Content Type" in prio.columns:
    prio["Is Event Cover"] = prio["Content Type"].astype(str).str.contains(r"\bevent\s*cover\b", case=False, na=False)
else:
    prio["Is Event Cover"] = False

# 5) Scoring: Overdue/near deadlines + Event Cover boost
def urgency_score(days):
    if pd.isna(days):
        return 0
    try:
        d = int(days)
    except Exception:
        return 0
    if d <= 0:   # overdue or due today
        return 100
    elif d <= 1:
        return 90
    elif d <= 3:
        return 80
    elif d <= 7:
        return 60
    elif d <= 14:
        return 40
    elif d <= 30:
        return 20
    else:
        return 0

prio["Urgency Score"] = prio["Days to Deadline"].apply(urgency_score)
EVENT_BOOST = 50  # tweak if you want
prio["Priority Score"] = prio["Urgency Score"] + prio["Is Event Cover"].astype(int) * EVENT_BOOST

# 6) Sort and show Top N
prio = prio.sort_values(by=["Priority Score", "Days to Deadline"], ascending=[False, True], na_position="last")
top_n = st.slider("Show Top N priority tasks", min_value=5, max_value=50, value=10, step=1)

# Columns to display
display_cols = []
for c in ["Content Title", "Designer Name", "Assigned By", "Content Type",
          "Assigned Date", "Completion Date"]:
    if c in prio.columns:
        display_cols.append(c)
# add chosen deadline column and computed fields
if deadline_col and deadline_col not in display_cols:
    display_cols.append(deadline_col)
display_cols += [c for c in ["Days to Deadline", "Is Event Cover", "Priority Score"] if c in prio.columns]

# Fallback if nothing matched
if not display_cols:
    display_cols = prio.columns.tolist()

st.dataframe(prio.head(top_n)[display_cols], use_container_width=True, hide_index=True)

    # 📊 Task Completion Overview
    st.subheader("📊 Task Completion Overview")
    status_counts = filtered["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig1 = px.bar(
        status_counts,
        x="Status",
        y="Count",
        color="Status",
        text="Count",
        color_discrete_map={"Completed": "green", "Remaining": "red"}
    )
    fig1.update_layout(xaxis_title="", yaxis_title="Tasks", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # 🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining
    # -----------------------------
    st.subheader("🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining")

    # Let "editor" be either Designer or Assigner
    compare_dim = st.selectbox(
        "Compare by",
        options=["Designer Name", "Assigned By"],
        index=0,
        help="Pick which role you consider as 'editor' for this comparison."
    )

    filtered_comp = filtered.copy()
    filtered_comp[compare_dim] = filtered_comp[compare_dim].fillna("Unknown")

    team_counts = (
        filtered_comp
        .groupby([compare_dim, "Status"])
        .size()
        .reset_index(name="Count")
    )

    all_editors = team_counts[compare_dim].unique()
    status_order = ["Completed", "Remaining"]
    full_index = pd.MultiIndex.from_product([all_editors, status_order], names=[compare_dim, "Status"])
    team_counts = (
        team_counts
        .set_index([compare_dim, "Status"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    pivot_for_sort = team_counts.pivot(index=compare_dim, columns="Status", values="Count").fillna(0)
    sorted_editors = pivot_for_sort.sort_values(by="Completed", ascending=False).index.tolist()
    n_editors = len(sorted_editors)

    if n_editors == 0:
        st.info("No editors to compare for the current filters.")
        keep_editors = set()
        team_counts_top = team_counts.iloc[0:0]
        top_n = 0
    else:
        if n_editors == 1:
            st.caption("Only one editor in the current filter. Showing all results.")
            top_n = 1
        else:
            # slider needs min < max; allow full dynamic range
            top_n = st.slider(
                "Show Top N",
                min_value=1,
                max_value=n_editors,
                value=min(10, n_editors),
                help="Limit how many editors to display for readability."
            )
        keep_editors = set(sorted_editors[:top_n])
        team_counts_top = team_counts[team_counts[compare_dim].isin(keep_editors)]

    if n_editors > 0:
        fig_team = px.bar(
            team_counts_top,
            y=compare_dim,
            x="Count",
            color="Status",
            orientation="h",
            category_orders={
                compare_dim: [e for e in sorted_editors if e in keep_editors],
                "Status": status_order
            },
            text="Count",
            barmode="group",
            color_discrete_map={"Completed": "green", "Remaining": "red"},
            height=max(400, 30 * len(keep_editors))
        )
        fig_team.update_layout(
            xaxis_title="Tasks",
            yaxis_title="",
            legend_title="",
            bargap=0.25,
            margin=dict(l=120, r=20, t=40, b=20),
        )
        fig_team.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_team, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**🏆 Top Completed**")
            top_completed = (
                pivot_for_sort["Completed"]
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
                .rename(columns={compare_dim: "Editor", "Completed": "Completed Tasks"})
            )
            st.dataframe(top_completed, use_container_width=True, hide_index=True)

        with colB:
            st.markdown("**⏳ Most Remaining**")
            top_remaining = (
                pivot_for_sort["Remaining"]
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
                .rename(columns={compare_dim: "Editor", "Remaining": "Remaining Tasks"})
            )
            st.dataframe(top_remaining, use_container_width=True, hide_index=True)

    # 📆 Assignment vs Completion Timeline
    st.subheader("📆 Assignment vs Completion Timeline")
    filtered = filtered.copy()
    filtered["Assigned Day"] = filtered["Assigned Date"].dt.date
    filtered["Completed Day"] = filtered["Completion Date"].dt.date

    assigned = filtered.groupby("Assigned Day").size().reset_index(name="Assigned")
    completed = filtered.groupby("Completed Day").size().reset_index(name="Completed")

    timeline = pd.merge(assigned, completed, left_on="Assigned Day", right_on="Completed Day", how="outer")
    timeline["Day"] = timeline["Assigned Day"].combine_first(timeline["Completed Day"])
    timeline = timeline.fillna(0).sort_values("Day")
    timeline = timeline[["Day", "Assigned", "Completed"]]

    fig2 = px.line(timeline, x="Day", y=["Assigned", "Completed"], markers=True)
    fig2.update_layout(yaxis_title="Tasks", height=450)
    st.plotly_chart(fig2, use_container_width=True)

    # 📂 Content Type Breakdown
    st.subheader("📂 Tasks by Content Type")
    content_counts = filtered["Content Type"].value_counts().reset_index()
    content_counts.columns = ["Content Type", "Count"]
    fig3 = px.bar(content_counts, y="Content Type", x="Count", orientation="h", text="Count", height=600)
    fig3.update_layout(yaxis_title="", xaxis_title="Tasks", margin=dict(l=100))
    st.plotly_chart(fig3, use_container_width=True)

    # 📅 Weekday Assignment Distribution
    st.subheader("📅 Tasks by Weekday")
    filtered["Weekday"] = filtered["Assigned Date"].dt.day_name()
    weekday_counts = (
        filtered["Weekday"]
        .value_counts()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .fillna(0)
        .reset_index()
    )
    weekday_counts.columns = ["Weekday", "Count"]
    fig4 = px.bar(weekday_counts, x="Weekday", y="Count", text="Count", color="Weekday", height=400)
    st.plotly_chart(fig4, use_container_width=True)

    # 📋 Task Tables
    table_cols = ["Content Title", "Designer Name", "Assigned By", "Assigned Date", "Completion Date", "Design Status"]
    st.subheader("📋 Completed Tasks")
    st.dataframe(filtered[filtered["Status"] == "Completed"][table_cols], use_container_width=True)

    st.subheader("📋 Remaining Tasks")
    st.dataframe(filtered[filtered["Status"] == "Remaining"][table_cols], use_container_width=True)

else:
    st.info("📌 Please upload a CSV file to begin.")
