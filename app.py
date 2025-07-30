import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import re
import gspread
from google.oauth2.service_account import Credentials
from dateutil import parser as dtparser



# --- Ensure unique, clean column names everywhere ---
def uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and make duplicate header names unique by appending .1, .2, ...
    e.g., 'Content Type', 'Content Type' -> 'Content Type', 'Content Type.1'
    """
    df = df.copy()
    raw = df.columns.astype(str)
    stripped = [c.strip() for c in raw]

    seen = {}
    new_cols = []
    for name in stripped:
        idx = seen.get(name, 0)
        new_cols.append(f"{name}.{idx}" if idx else name)
        seen[name] = idx + 1

    df.columns = new_cols
    return df


st.set_page_config(layout="wide")

st.title("📋 Employee Task Dashboard (Interactive with Plotly)")

# ========= EDIT THESE TWO (or three) CONSTANTS =========
JSON_PATH = None # on Streamlit Cloud we’ll use st.secrets instead of a file
SHEET_URL = "https://docs.google.com/spreadsheets/d/1yaW7V7hSBqOBZYbqIUKsGrhB8pVtWWrzq7t5scq3JVI/edit?gid=0#gid=0"  # your Google Sheet link
SHEET_GID = "0"        # optional: keep "0" or set to the gid of the tab you want
SHEET_TITLE = None     # optional alternative to GID: e.g., "Data". If set, it takes priority over GID.
# =======================================================

# Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

@st.cache_data(ttl=300)
def load_sheet_from_gdrive(json_path: str | None, sheet_url: str, sheet_gid: str | None = None, sheet_title: str | None = None) -> pd.DataFrame:
    """Load a private Google Sheet using Streamlit Secrets if available; otherwise a local JSON file."""
    # --- Build credentials ---
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        # Convert literal '\n' to real newlines if needed
        pk = info.get("private_key", "")
        if "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        if not json_path:
            raise ValueError("No credentials found: set JSON_PATH or define [gcp_service_account] in Streamlit Secrets.")
        creds = Credentials.from_service_account_file(json_path, scopes=SCOPES)

    gc = gspread.authorize(creds)

    # --- The rest of your original code stays the same ---
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        raise ValueError("Invalid Google Sheet URL")
    sheet_id = m.group(1)

    sh = gc.open_by_key(sheet_id)

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
    df = uniquify_columns(df)  # <-- ensures headers are unique and stripped

    return df

# ---------- Robust date parser ----------
def parse_date(x):
    """Robust date parser for CSV/Google Sheet cells."""
    if x is None:
        return pd.NaT
    if isinstance(x, float) and pd.isna(x):
        return pd.NaT
    if isinstance(x, str) and x.strip() == "":
        return pd.NaT

    if isinstance(x, (pd.Timestamp, datetime.datetime)):
        ts = pd.to_datetime(x, errors="coerce")
        return ts.tz_localize(None) if isinstance(ts, pd.Timestamp) and ts.tz is not None else ts
    if isinstance(x, datetime.date):
        return pd.Timestamp(x)

    # Excel/Sheets serial number (days since 1899-12-30)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            val = float(x)
            if 20000 <= val <= 60000:
                base = datetime.datetime(1899, 12, 30)
                return pd.to_datetime(base + datetime.timedelta(days=val))
        except Exception:
            pass

    s = str(x).strip()
    if s.lower() in {"na", "n/a", "none", "null", "-", "—", "tbd", "not set"}:
        return pd.NaT
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(\d{1,2})\s*[-–—]\s*\d{1,2}(\b| )", r"\1 ", s)

    today = datetime.date.today()
    default_dt = datetime.datetime(today.year, today.month, today.day)

    for try_fn in (
        lambda v: pd.to_datetime(v, dayfirst=True, errors="raise"),
        lambda v: pd.to_datetime(v, errors="raise"),
        lambda v: dtparser.parse(v, dayfirst=True, fuzzy=True, default=default_dt),
    ):
        try:
            ts = try_fn(s)
            if isinstance(ts, pd.Timestamp):
                return ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts
            return pd.to_datetime(ts)
        except Exception:
            continue
    return pd.NaT

# ----------------------------
# Helpers for date defaults
# ----------------------------
def _ensure_datetime_col(df_in: pd.DataFrame, col: str) -> pd.Series:
    """Return a datetime64 series for col (parsing if needed); or an empty series if col missing."""
    if col not in df_in.columns:
        return pd.Series(dtype="datetime64[ns]")
    s = df_in[col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = s.apply(parse_date)
    return pd.to_datetime(s, errors="coerce")

def _minmax_over_cols(df_in: pd.DataFrame, cols: list[str]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Get the earliest min and latest max across multiple date columns."""
    mins, maxs = [], []
    for c in cols:
        s = _ensure_datetime_col(df_in, c)
        if not s.empty:
            vmin, vmax = s.min(skipna=True), s.max(skipna=True)
            if pd.notna(vmin): mins.append(vmin)
            if pd.notna(vmax): maxs.append(vmax)
    return (min(mins) if mins else pd.NaT, max(maxs) if maxs else pd.NaT)

# ---------- Data source: CSV upload OR Google Sheet ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = uniquify_columns(df)  # <-- make headers unique
else:
    with st.spinner("Loading live Google Sheet…"):
        df = load_sheet_from_gdrive(JSON_PATH, SHEET_URL, sheet_gid=SHEET_GID, sheet_title=SHEET_TITLE)
        df = uniquify_columns(df)  # <-- make headers unique

# If still no data, stop
if df is None or df.empty:
    st.info("No data found. Upload a CSV or check your Google Sheet link/permissions.")
    st.stop()

# ---------- Clean & prepare ----------
df.columns = df.columns.str.strip()

# Drop unnecessary columns
drop_cols = [col for col in df.columns if "Unnamed" in col or "Design Drive Link" in col]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Remove month label rows like "April 2025"
if "Content Type" in df.columns:
    df = df[df["Content Type"].notna()]
    df = df[~df["Content Type"].astype(str).str.contains(r"20[0-9]{2}", na=False)]

# Parse dates and clean status
if "Assigned Date" in df.columns:
    df["Assigned Date"] = df["Assigned Date"].apply(parse_date)
if "Completion Date" in df.columns:
    df["Completion Date"] = df["Completion Date"].apply(parse_date)

if "Design Status" in df.columns:
    df["Design Status"] = df["Design Status"].fillna("Non Completed").astype(str).str.strip()
else:
    df["Design Status"] = df.get("Status", "Non Completed").fillna("Non Completed").astype(str).str.strip()

# Business rule: exactly "completed" => Completed, everything else => Remaining
df["Status"] = df["Design Status"].apply(
    lambda x: "Completed" if str(x).strip().lower() == "completed" else "Remaining"
)

# ---------- Sidebar filters ----------
designers = ["All"] + (sorted(df["Designer Name"].dropna().unique()) if "Designer Name" in df.columns else [])
assigners = ["All"] + (sorted(df["Assigned By"].dropna().unique()) if "Assigned By" in df.columns else [])
selected_designer = st.sidebar.selectbox("🎨 Filter by Designer", designers, index=0 if designers else None)
selected_by = st.sidebar.selectbox("🧑‍💼 Filter by Assigned By", assigners, index=0 if assigners else None)

# ----------------------------
# PRIMARY DATE RANGE (Assigned Date) — defaults to first/last date found in the sheet
# ----------------------------
sheet_min_assigned, sheet_max_assigned = _minmax_over_cols(df, ["Assigned Date"])
# If Assigned Date is missing, fall back to earliest among Deadline/Completion Date
if pd.isna(sheet_min_assigned) or pd.isna(sheet_max_assigned):
    sheet_min_any, sheet_max_any = _minmax_over_cols(df, ["Assigned Date", "Deadline", "Completion Date"])
    default_start_primary = (sheet_min_any if pd.notna(sheet_min_any) else pd.Timestamp(datetime.date.today()))
    default_end_primary = (sheet_max_any if pd.notna(sheet_max_any) else pd.Timestamp(datetime.date.today()))
else:
    default_start_primary = sheet_min_assigned
    default_end_primary = sheet_max_assigned

# Convert defaults to date objects for Streamlit control
PRIMARY_DEFAULT_START = default_start_primary.date()
PRIMARY_DEFAULT_END = default_end_primary.date()

quick_period = st.sidebar.radio(
    "⏱️ Quick period",
    ["All time", "This month", "Last 30 days", "This year"],
    index=0,
    help="Fast presets for the primary date range."
)

# Compute quick-period defaults
# Compute quick-period defaults
today = datetime.date.today()
if quick_period == "All time":
    primary_default_value = (PRIMARY_DEFAULT_START, PRIMARY_DEFAULT_END)
elif quick_period == "This month":
    start_m = today.replace(day=1)
    next_m = (datetime.date(today.year + (today.month == 12), 1 if today.month == 12 else today.month + 1, 1))
    end_m = next_m - datetime.timedelta(days=1)
    primary_default_value = (start_m, end_m)
elif quick_period == "Last 30 days":
    primary_default_value = (today - datetime.timedelta(days=30), today)
else:  # This year
    start_y = datetime.date(today.year, 1, 1)
    end_y = datetime.date(today.year, 12, 31)
    primary_default_value = (start_y, end_y)

# --- Force default to start at Jan 1, 2024 (keeps your dynamic end date) ---
primary_default_value = (datetime.date(2025, 1, 1), PRIMARY_DEFAULT_END)

primary_date_from, primary_date_to = st.sidebar.date_input(
    "📅 Primary date range — by Assigned Date",
    value=primary_default_value,
    help="This range filters the dashboard by Assigned Date (defaults to 2024-01-01 to the latest date in the sheet by default).",
)

# Apply Designer/Assigner filters first
filtered = df.copy()
if "Designer Name" in filtered.columns and selected_designer and selected_designer != "All":
    filtered = filtered[filtered["Designer Name"] == selected_designer]
if "Assigned By" in filtered.columns and selected_by and selected_by != "All":
    filtered = filtered[filtered["Assigned By"] == selected_by]

# Ensure Assigned Date is datetime (idempotent)
if "Assigned Date" in filtered.columns:
    filtered["Assigned Date"] = _ensure_datetime_col(filtered, "Assigned Date")

# Apply primary date filter
if "Assigned Date" in filtered.columns and primary_date_from and primary_date_to:
    filtered = filtered[
        (filtered["Assigned Date"] >= pd.to_datetime(primary_date_from)) &
        (filtered["Assigned Date"] <= pd.to_datetime(primary_date_to))
    ]

# ----------------------------
# Global filters (useful extras)
# ----------------------------
# 1) Global Content Type filter (defaults to SM post + Event Cover; has 'All' and '(blank)')
def apply_ct_filter_ui(label: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shows a multiselect for Content Type and returns the filtered df.
    - 'All' option (no filtering when selected)
    - '(blank)' option for empty/NaN
    - Default preselects 'SM post' and 'Event Cover' (case-insensitive) if present; otherwise 'All'
    """
    if "Content Type" not in base_df.columns:
        st.caption("No 'Content Type' column found.")
        return base_df

    ct_raw = base_df["Content Type"].astype(str).str.strip()
    ct_lower = base_df["Content Type"].fillna("").astype(str).str.strip().str.lower()

    uniq_display = sorted([v for v in ct_raw.unique() if v != ""])
    has_blank = (ct_lower == "").any()

    options = ["All"] + uniq_display + (["(blank)"] if has_blank else [])

    desired_defaults_lower = {"All"}
    present_defaults = [v for v in uniq_display if v.lower() in desired_defaults_lower]
    default_selection = present_defaults if present_defaults else ["All"]

    selected = st.sidebar.multiselect(label, options, default=default_selection)

    # If 'All' is selected OR nothing selected -> return unfiltered base_df
    if (not selected) or ("All" in selected):
        return base_df

    # Build mask
    mask = pd.Series(False, index=base_df.index)

    # Named selections (case-insensitive)
    named = [s for s in selected if s not in {"All", "(blank)"}]
    if named:
        named_lower = set([s.lower() for s in named])
        mask = mask | ct_lower.isin(named_lower)

    # Blanks
    if "(blank)" in selected:
        mask = mask | (ct_lower == "")

    return base_df[mask]

filtered = apply_ct_filter_ui("🗂️ Global Content Type filter", filtered)

# 2) Quick title search (case-insensitive contains)
title_query = st.sidebar.text_input("🔎 Search in Content Title", value="")
if title_query and "Content Title" in filtered.columns:
    mask_title = filtered["Content Title"].astype(str).str.contains(title_query, case=False, na=False)
    filtered = filtered[mask_title]

# Early exit if no rows after global filters
if filtered.empty:
    st.warning("No data for the selected filters/time frame. Try widening the date range, clearing the title search, or picking 'All' in Content Type.")
    st.stop()

# ----------------------------
# SECONDARY DATE RANGE (independent, optional)
# ----------------------------
with st.expander("🔁 Optional secondary date filter (independent)", expanded=False):
    secondary_field = st.selectbox(
        "Filter additionally by which date column?",
        options=["None", "Assigned Date", "Deadline", "Completion Date"],
        index=0,
        help="Choose a date column for an extra filter. Leave as 'None' to skip."
    )

    if secondary_field != "None":
        col_series = _ensure_datetime_col(filtered, secondary_field)
        if col_series.empty or col_series.dropna().empty:
            st.info(f"No usable dates in '{secondary_field}' for the current filters; secondary filter skipped.")
        else:
            sec_min = col_series.min(skipna=True).date()
            sec_max = col_series.max(skipna=True).date()

            sec_from, sec_to = st.date_input(
                f"Secondary date range — by {secondary_field}",
                value=(sec_min, sec_max),
            )

            filtered = filtered[
                (col_series >= pd.to_datetime(sec_from)) &
                (col_series <= pd.to_datetime(sec_to))
            ]

# Check again after secondary
if filtered.empty:
    st.warning("No data after applying the secondary date filter. Try widening the range or switching the date column.")
    st.stop()

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

# ======================
# 🏆 Employee of the Month  (Completed-only + monthly insights)
# ======================
st.subheader("🏆 Employee of the Month")

# Only show if ALL designers are selected (irrelevant when a single designer is chosen)
if selected_designer == "All":
    # Start from the currently filtered dataset
    eom_df = filtered.copy()

    # 1) Restrict to COMPLETED tasks only (prefer normalized Status)
    status_col = "Status" if "Status" in eom_df.columns else ("Design Status" if "Design Status" in eom_df.columns else None)
    if status_col is not None:
        eom_df = eom_df[eom_df[status_col].astype(str).str.strip().str.lower() == "completed"]
    else:
        st.info("No status column found to restrict scoring to completed tasks.")
        eom_df = eom_df.iloc[0:0]

    # If no completed tasks, show friendly message and bail out early
    if eom_df.empty or "Assigned Date" not in eom_df.columns:
        st.info("No completed tasks in the current filters/time window to compute Employee of the Month.")
    else:
        # 2) Build a clean month dropdown from ASSIGNED DATE, remove future months, dedupe, nice labels
        assigned_series = pd.to_datetime(eom_df["Assigned Date"], errors="coerce")
        month_periods = assigned_series.dropna().dt.to_period("M")
        if month_periods.empty:
            st.info("No valid Assigned Dates found to build the month selector.")
        else:
            # Hide any future month (e.g., accidental future data)
            current_month = pd.Period(pd.Timestamp.today(), freq="M")
            month_periods = month_periods[month_periods <= current_month]

            # Unique, sorted (ascending), then build readable labels
            unique_months = sorted(month_periods.unique())
            month_keys = [str(p) for p in unique_months]  # "YYYY-MM"
            month_labels = [p.to_timestamp().strftime("%B %Y") for p in unique_months]
            key_by_label = dict(zip(month_labels, month_keys))

            # Final selectbox options
            month_label_selected = st.selectbox("📅 Select month", ["All"] + month_labels, index=0)

            # Apply month filter if not "All"
            month_scope_df = eom_df.copy()
            if month_label_selected != "All":
                sel_key = key_by_label[month_label_selected]  # "YYYY-MM"
                month_scope_df = month_scope_df[
                    pd.to_datetime(month_scope_df["Assigned Date"], errors="coerce").dt.to_period("M").astype(str) == sel_key
                ]

            # If still empty after month filter
            if month_scope_df.empty:
                st.info("No completed tasks for the chosen month.")
            else:
                # 3) Points map (category → points); fallback 1 point for anything not listed
                points_map = {
                    "branding": 10, "video": 10, "reel": 10, "standee": 8,
                    "event cover": 4, "banners": 4, "carousel": 8, "paid ads": 7,
                    "newsletter": 8, "logo": 14, "dp": 1, "blogs": 3, "card": 8,
                    "reel graphics": 3, "sm post": 3
                }
                # robust matching that uses substrings (e.g., "SM post – static")
                keys_sorted = sorted(points_map.keys(), key=len, reverse=True)

                def _type_key(val: str) -> str:
                    s = (str(val) if pd.notna(val) else "").strip().lower()
                    if not s:
                        return "other"
                    for k in keys_sorted:
                        if k in s:
                            return k
                    return "other"

                # Normalize type and compute points-per-task & row points (1 row = 1 task)
                month_scope_df = month_scope_df.copy()
                month_scope_df["TypeKey"] = month_scope_df.get("Content Type", "").apply(_type_key)
                month_scope_df["PtsPerTask"] = month_scope_df["TypeKey"].map(points_map).fillna(1).astype(int)
                month_scope_df["Points"] = month_scope_df["PtsPerTask"]

                # 4) Leaderboard (sum of points per designer) — used for medals & order
                if "Designer Name" not in month_scope_df.columns:
                    st.info("No 'Designer Name' column found.")
                else:
                    emp_points = (
                        month_scope_df.groupby("Designer Name")["Points"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
                    )

                    if emp_points.empty:
                        st.info("No completed tasks to score for the selected month.")
                    else:
                        # Top 3 with medals
                        medals = ["🥇", "🥈", "🥉"]
                        top3 = emp_points.head(3)
                        for i, row in top3.iterrows():
                            medal = medals[i] if i < len(medals) else "🏅"
                            st.markdown(f"{medal} **{row['Designer Name']}** — **{int(row['Points'])} pts**")

                        # --- Aggregation for stacked vertical bars (designer x type) ---
                        viz_df = month_scope_df.copy()
                        agg = (
                            viz_df
                            .groupby(["Designer Name", "TypeKey", "PtsPerTask"], dropna=False)
                            .agg(Tasks=("TypeKey", "count"),
                                 Points=("Points", "sum"))
                            .reset_index()
                        )

                        # Order designers by total points (desc)
                        designer_totals = agg.groupby("Designer Name")["Points"].sum().sort_values(ascending=False)
                        designer_order = designer_totals.index.tolist()
                        agg["Designer Name"] = pd.Categorical(agg["Designer Name"], categories=designer_order, ordered=True)

                        # Legend labels like "sm post (3)"
                        def _label_with_points(row):
                            tk = row["TypeKey"] if row["TypeKey"] else "other"
                            pts = int(row["PtsPerTask"])
                            return f"{tk} ({pts})"

                        agg["TypeLabel"] = agg.apply(_label_with_points, axis=1)

                        # Sort legend entries by total points across all designers (desc)
                        type_totals = agg.groupby("TypeLabel")["Points"].sum().sort_values(ascending=False)
                        type_order = type_totals.index.tolist()

                        # --- Color mapping (consistent between chart and custom legend) ---
                        palette = px.colors.qualitative.D3  # choose any Plotly qualitative palette
                        color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(type_order)}

                        # --- Build stacked vertical bar chart (disable built-in legend) ---
                        import plotly.express as px  # ensure px is in scope in this block
                        fig_emp = px.bar(
                            agg,
                            x="Designer Name",
                            y="Points",
                            color="TypeLabel",
                            category_orders={
                                "Designer Name": designer_order,
                                "TypeLabel": type_order,
                            },
                            color_discrete_map=color_map,
                            barmode="stack",
                            text="Points",
                        )

                        # Rich hover: Designer → Type → Tasks → Pts/task → Subtotal → Designer total
                        designer_total_map = designer_totals.to_dict()
                        agg["_DesignerTotal"] = agg["Designer Name"].map(designer_total_map)
                        fig_emp.update_traces(
                            customdata=list(zip(agg["Tasks"], agg["PtsPerTask"], agg["_DesignerTotal"])),
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "%{trace.name}<br>"             # e.g., "sm post (3)"
                                "Tasks: %{customdata[0]}<br>"
                                "Pts per task: %{customdata[1]}<br>"
                                "Subtotal points: %{y}<br>"
                                "Designer total: %{customdata[2]}"
                                "<extra></extra>"
                            ),
                            textposition="outside",
                            cliponaxis=False,
                            showlegend=False,  # we'll render our own legend in the right column
                        )

                        # Chart height (used for legend container too)
                        chart_height = max(380, 26 * max(1, len(designer_order)))

                        fig_emp.update_layout(
                            height=chart_height,
                            margin=dict(l=10, r=10, t=30, b=60),
                            xaxis_title="",
                            yaxis_title="Points",
                            showlegend=False,
                        )

                        # --- Two-column layout: chart (left) + custom legend (right, scrollable) ---
                        c_chart, c_legend = st.columns([0.74, 0.26])  # adjust ratios if needed

                        with c_chart:
                            st.plotly_chart(fig_emp, use_container_width=True)

                        with c_legend:
                            # Scrollable legend aligned to chart height
                            legend_inner_height = chart_height - 44  # leave room for header/padding
                            if legend_inner_height < 220:
                                legend_inner_height = 220  # minimum sensible height

                            st.markdown("**Legend — Type (points)**")
                            legend_html = f"""
                            <div style="
                                height:{legend_inner_height}px;
                                overflow-y:auto;
                                border:1px solid rgba(0,0,0,0.08);
                                border-radius:10px;
                                padding:8px 10px;
                                background:rgba(255,255,255,0.9);
                            ">
                                <div style='display:flex;flex-direction:column;gap:6px;'>
                            """
                            for lbl in type_order:
                                color = color_map.get(lbl, "#888")
                                legend_html += (
                                    "<div style='display:flex;align-items:center;'>"
                                    f"<span style='display:inline-block;width:12px;height:12px;"
                                    f"background:{color};border-radius:2px;margin-right:8px;flex:0 0 auto;'></span>"
                                    f"<span style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{lbl}</span>"
                                    "</div>"
                                )
                            legend_html += "</div></div>"
                            st.markdown(legend_html, unsafe_allow_html=True)
                            st.caption("Hover a bar to see tasks × points and subtotals.")

                        # 5) Insights (line bullets)
                        st.markdown("**🔎 Monthly insights (completed only)**")
                        lines = []

                        # Overall: top scorer
                        top_name = top3.iloc[0]["Designer Name"]
                        top_pts = int(top3.iloc[0]["Points"])
                        lines.append(f"- **Top scorer**: {top_name} with **{top_pts} points**.")

                        # Overall: most common category by count (within completed scope)
                        ct_counts = (
                            month_scope_df.groupby("TypeKey")
                            .size()
                            .sort_values(ascending=False)
                        )
                        if not ct_counts.empty:
                            lines.append(f"- **Most common task type**: {ct_counts.index[0]} (**{int(ct_counts.iloc[0])}** tasks).")

                        # Overall: highest-scoring category (sum of points by type)
                        ct_points = (
                            month_scope_df.groupby("TypeKey")["Points"]
                            .sum()
                            .sort_values(ascending=False)
                        )
                        if not ct_points.empty:
                            lines.append(f"- **Highest scoring category**: {ct_points.index[0]} (**{int(ct_points.iloc[0])}** pts).")

                        # Per-designer: their most-done category (by count), show a few lines
                        per_designer_ct = (
                            month_scope_df.groupby(["Designer Name", "TypeKey"])
                            .size()
                            .reset_index(name="Count")
                        )
                        # Choose up to 5 designers to summarize (sorted by total points)
                        topN = min(5, len(emp_points))
                        top_designers = emp_points["Designer Name"].head(topN).tolist()

                        for name in top_designers:
                            sub = per_designer_ct[per_designer_ct["Designer Name"] == name]
                            if sub.empty:
                                continue
                            idx = sub["Count"].idxmax()
                            best_ct = sub.loc[idx, "TypeKey"]
                            best_ct_n = int(sub.loc[idx, "Count"])
                            total_pts = int(emp_points.loc[emp_points["Designer Name"] == name, "Points"].iloc[0])
                            lines.append(f"- **{name}** did the most **{best_ct}** (**{best_ct_n}** tasks; **{total_pts} pts**).")

                        for ln in lines:
                            st.markdown(ln)
else:
    st.caption("👤 Employee of the Month is hidden when a single designer is selected.")



# ---- Export (useful extra) ----
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv, file_name="filtered_tasks.csv", mime="text/csv")

# =========================
# 🔥 Priority Tasks (Deadline + Event Cover, normalized)
# =========================
st.subheader("🔥 Priority Tasks")

# Start from the filtered view; only show remaining (not completed)
prio = filtered.copy()
if "Status" in prio.columns:
    prio = prio[prio["Status"] != "Completed"]

# Ensure Deadline is datetime and compute days-to-deadline
# --- Safer Days-to-Deadline computation ---
if "Deadline" in prio.columns:
    # Force proper datetimes no matter what came in
    prio["Deadline"] = pd.to_datetime(prio["Deadline"].apply(parse_date), errors="coerce")
    today_midnight = pd.Timestamp(datetime.date.today())  # normalized (00:00)
    prio["Days to Deadline"] = (prio["Deadline"] - today_midnight).dt.days
else:
    prio["Days to Deadline"] = pd.NA


# Normalize content type and flag "Event Cover"
if "Content Type" in prio.columns:
    ct_norm = (
        prio["Content Type"].astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    prio["Is Event Cover"] = ct_norm.str.contains(r"\bevent cover\b", na=False)
else:
    prio["Is Event Cover"] = False

# Urgency scoring
def _urgency(days):
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

# Interactive controls
c_ctrl1, c_ctrl2 = st.columns([1, 1])
with c_ctrl1:
    top_n_prio = st.slider("Show Top N priority tasks", min_value=5, max_value=50, value=10, step=1)
with c_ctrl2:
    event_boost = st.slider("Event Cover boost", min_value=0, max_value=100, value=50, step=5,
                            help="Extra priority for tasks whose Content Type is Event Cover (normalized).")

prio["Urgency Score"] = prio["Days to Deadline"].apply(_urgency)
prio["Priority Score"] = prio["Urgency Score"] + prio["Is Event Cover"].astype(int) * event_boost

# Sort: higher priority first; tie-breaker: closer deadline
prio = prio.sort_values(by=["Priority Score", "Days to Deadline"], ascending=[False, True], na_position="last")

# Compact display title
if "Content Title" in prio.columns:
    prio["Title (short)"] = prio["Content Title"].astype(str).str.slice(0, 60).where(
        prio["Content Title"].astype(str).str.len() <= 60,
        prio["Content Title"].astype(str).str.slice(0, 57) + "…"
    )

# Columns for the table
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

# Layout: table + compact bar chart
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
            show_chart,
            y=y_field,
            x="Priority Score",
            orientation="h",
            color="Is Event Cover",
            color_discrete_map={True: "#d62728", False: "#1f77b4"},
            text="Priority Score",
            hover_data=[c for c in ["Deadline", "Days to Deadline", "Content Type"] if c in show_chart.columns],
            height=max(350, 26 * len(show_chart))
        )
        fig_prio.update_layout(
            xaxis_title="Priority",
            yaxis_title="",
            legend_title="Event Cover",
            margin=dict(l=10, r=10, t=10, b=10),
            bargap=0.25,
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
    status_counts,
    x="Status",
    y="Count",
    color="Status",
    text="Count",
    color_discrete_map={"Completed": "green", "Remaining": "red"}
)
fig1.update_layout(xaxis_title="", yaxis_title="Tasks", height=400)
st.plotly_chart(fig1, use_container_width=True)

# 🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining
st.subheader("🧑‍🤝‍🧑 Team Comparison — Completed vs Remaining")
compare_dim = st.selectbox(
    "Compare by",
    options=[c for c in ["Designer Name", "Assigned By"] if c in filtered.columns],
    index=0 if any(c in filtered.columns for c in ["Designer Name", "Assigned By"]) else None,
    help="Pick which role you consider as 'editor' for this comparison."
)

if compare_dim:
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
        team_counts_top = team_counts.iloc[0:0]
        top_n = 0
        keep_editors = set()
    else:
        if n_editors == 1:
            st.caption("Only one editor in the current filter. Showing all results.")
            top_n = 1
        else:
            top_n = st.slider("Show Top N", min_value=1, max_value=n_editors, value=min(10, n_editors),
                              help="Limit how many editors to display for readability.")
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
        filtered["Weekday"]
        .value_counts()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .fillna(0)
        .reset_index()
    )
    weekday_counts.columns = ["Weekday", "Count"]
    fig4 = px.bar(weekday_counts, x="Weekday", y="Count", text="Count", color="Weekday", height=400)
    st.plotly_chart(fig4, use_container_width=True)

# 📋 Task Tables (with per-table Content Type filters)
table_cols = [c for c in ["Content Title", "Designer Name", "Assigned By", "Assigned Date", "Completion Date", "Design Status", "Content Type"] if c in filtered.columns]
if table_cols:
    st.subheader("📋 Completed Tasks")
    completed_df = filtered[filtered["Status"] == "Completed"].copy()

    # Reuse the same CT filter but per table (UI appears inline, not sidebar)
    def apply_ct_filter_ui_inline(label: str, base_df: pd.DataFrame) -> pd.DataFrame:
        if "Content Type" not in base_df.columns:
            st.caption("No 'Content Type' column found.")
            return base_df
        ct_raw = base_df["Content Type"].astype(str).str.strip()
        ct_lower = base_df["Content Type"].fillna("").astype(str).str.strip().str.lower()
        uniq_display = sorted([v for v in ct_raw.unique() if v != ""])
        has_blank = (ct_lower == "").any()
        options = ["All"] + uniq_display + (["(blank)"] if has_blank else [])
        desired_defaults_lower = {"sm post", "event cover"}
        present_defaults = [v for v in uniq_display if v.lower() in desired_defaults_lower]
        default_selection = present_defaults if present_defaults else ["All"]
        selected = st.multiselect(label, options, default=default_selection)
        if (not selected) or ("All" in selected):
            return base_df
        mask = pd.Series(False, index=base_df.index)
        named = [s for s in selected if s not in {"All", "(blank)"}]
        if named:
            named_lower = set([s.lower() for s in named])
            mask = mask | ct_lower.isin(named_lower)
        if "(blank)" in selected:
            mask = mask | (ct_lower == "")
        return base_df[mask]

    completed_df = apply_ct_filter_ui_inline("Filter Content Type (Completed)", completed_df)
    if completed_df.empty:
        st.info("No completed tasks for the selected Content Type(s).")
    else:
        st.dataframe(completed_df[table_cols], use_container_width=True)

    st.subheader("📋 Remaining Tasks")
    remaining_df = filtered[filtered["Status"] == "Remaining"].copy()
    remaining_df = apply_ct_filter_ui_inline("Filter Content Type (Remaining)", remaining_df)
    if remaining_df.empty:
        st.info("No remaining tasks for the selected Content Type(s).")
    else:
        st.dataframe(remaining_df[table_cols], use_container_width=True)

# ======================
# 👨🏻‍💻 Developer Footer (pretty & interactive)
# ======================
from streamlit.components.v1 import html as st_html

DEV_NAME     = "Sameer Raza Malik"
DEV_EMAIL    = "sameer.raza@live.com"
DEV_LINKEDIN = "https://www.linkedin.com/in/sameer-raza-malik-586829361/"
# Optionally add more links here:
DEV_LINKS = {
    "LinkedIn": DEV_LINKEDIN,
    # "GitHub": "https://github.com/your-handle",
    # "Website": "https://your-site.com",
}

def render_dev_footer(
    name: str,
    email: str,
    links: dict[str, str],
    *,
    sticky: bool = False,   # set True to keep it always visible
    height: int = 140
) -> None:
    position = "fixed" if sticky else "relative"
    bottom_val = "0" if sticky else "auto"
    box_shadow = "0 -8px 20px rgba(0,0,0,0.06)" if sticky else "0 8px 20px rgba(0,0,0,0.06)"

    # Build link pills
    link_items = ""
    for label, url in links.items():
        if not url:
            continue
        # choose an emoji based on label (simple & lightweight)
        icon = "🔗"
        if "link" in label.lower():
            icon = "🔗"
        if "github" in label.lower():
            icon = "🐙"
        if "web" in label.lower() or "site" in label.lower():
            icon = "🌐"
        link_items += f"""
            <a class="pill" href="{url}" target="_blank" rel="noopener noreferrer" title="{label}">
                <span class="dot"></span>{icon}&nbsp;{label}
            </a>
        """

    html_code = f"""
    <div class="dev-footer-wrap" role="contentinfo">
      <div class="dev-footer">
        <div class="left">
          <div class="avatar" aria-hidden="true">👨🏻‍💻</div>
          <div class="meta">
            <div class="name">{name}</div>
            <div class="role">Developer</div>
            <div class="email-row">
              <span class="label">Email</span>
              <a class="email" href="mailto:{email}">{email}</a>
              <button class="copy" onclick="copyEmail()" title="Copy email">Copy</button>
            </div>
          </div>
        </div>
        <div class="right">
          <div class="links">{link_items}</div>
        </div>
      </div>
    </div>

    <style>
      .dev-footer-wrap {{
        position: {position};
        left: 0; right: 0; bottom: {bottom_val};
        z-index: 999;
      }}
      .dev-footer {{
        margin: 18px 0 0 0;
        padding: 14px 16px;
        background: linear-gradient(180deg, #f7faff 0%, #eef4ff 50%, #e9f7ff 100%);
        border-top: 1px solid rgba(0,0,0,0.06);
        box-shadow: {box_shadow};
      }}
      /* content container */
      .dev-footer {{
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 14px;
      }}
      .left {{
        display: grid;
        grid-template-columns: 56px 1fr;
        gap: 12px;
        align-items: center;
      }}
      .avatar {{
        width: 56px; height: 56px;
        display: grid; place-items: center;
        font-size: 26px;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.06);
      }}
      .name {{
        font-weight: 700; font-size: 16px; color: #0f172a;
      }}
      .role {{
        margin-top: 2px; color: #475569; font-size: 12px;
      }}
      .email-row {{
        margin-top: 6px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      }}
      .label {{
        font-size: 11px; color: #64748b; background: #f1f5f9; padding: 3px 8px; border-radius: 999px;
      }}
      .email {{
        font-size: 13px; color: #0ea5e9; text-decoration: none;
      }}
      .email:hover {{ text-decoration: underline; }}

      .copy {{
        font-size: 12px; padding: 6px 10px; border-radius: 8px;
        border: 1px solid rgba(14,165,233,0.5); background: #e0f2fe; color: #0369a1;
        cursor: pointer;
      }}
      .copy:hover {{ background: #bae6fd; }}

      .right .links {{
        display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;
      }}
      .pill {{
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 12px; color: #0b1324; text-decoration: none;
        background: #ffffff; padding: 6px 10px; border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.08);
        white-space: nowrap;
      }}
      .pill:hover {{ border-color: rgba(0,0,0,0.2); }}
      .pill .dot {{
        width: 8px; height: 8px; border-radius: 999px; background: #60a5fa;
        box-shadow: 0 0 0 2px rgba(96,165,250,0.25);
        margin-right: 2px;
      }}

      /* Responsive */
      @media (max-width: 800px) {{
        .dev-footer {{ grid-template-columns: 1fr; }}
        .right .links {{ justify-content: flex-start; }}
      }}
    </style>

    <script>
      function copyEmail() {{
        const email = "{email}";
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(email).then(() => setCopied());
        }} else {{
          const ta = document.createElement('textarea');
          ta.value = email; document.body.appendChild(ta);
          ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
          setCopied();
        }}
      }}
      function setCopied() {{
        const btn = document.querySelector('.copy');
        if (!btn) return;
        const old = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => btn.textContent = "Copy", 1200);
      }}
    </script>
    """
    # If sticky, give the iframe enough height to avoid covering content
    st_html(html_code, height=height)

# 👉 Render it (place this near the end of your app)
render_dev_footer(DEV_NAME, DEV_EMAIL, DEV_LINKS, sticky=False, height=150)
