# 📊 FES Pulse — README

**FES Pulse** is a powerful Streamlit-based productivity dashboard designed to visualize, filter, and analyze team task data directly from a Google Sheet. It gives both high-level summaries and deep dives into individual and team performance.

---

---

## 🔗 App Link

You can access the live app here: [https://fes-eda.streamlit.app/](https://fes-eda.streamlit.app/)

---

## 🌟 Features Overview

### 🔍 Global Filters

* **Primary Date Range Filter**: Filters tasks based on `Assigned Date`.
* **Global Content Type Filter**: Sidebar multiselect with options like "All", specific content types, and "(blank)".
* **Title Search**: Case-insensitive keyword search within `Content Title`.
* **Secondary Date Filter** *(optional)*: Independently filter by `Assigned Date`, `Deadline`, or `Completion Date`.

---

## 🧮 Key KPIs

* **📌 Total Tasks**
* **✅ Completed**
* **❌ Remaining**
* **🔄 Completion Progress Bar**

---

## 🏆 Monthly Recognition

* **Designer of the Month / Assigner of the Month** based on a custom point system per task type.

* **Points System** (examples):

  * SM Post: 3 pts
  * Logo: 14 pts
  * Event Cover: 4 pts
  * Branding: 10 pts
  * Others default to 1 pt

* **Features:**

  * Select month view or aggregate all months.
  * Toggle between designer and assigner roles.
  * Stacked bar chart showing contributions by task type.
  * Hover tooltips show task count, points per task, and total points.
  * Expandable individual breakdowns (if "Assigner of the Month" is selected).

---

## 📋 Individual Task Breakdown

* Shows a detailed breakdown of completed tasks per person.
* Grouped by task type with total points, points per task, and task count.
* Fully respects global filters.

---

## 📊 Task Completion Overview

* Horizontal bar chart showing total number of tasks grouped by status (Completed, Remaining).

---

## 🔥 Priority Tasks

* Highlights most urgent tasks based on:

  * `Deadline` proximity
  * Content type (event covers receive extra weight)
* **Controls**:

  * Slider to adjust how many top priority tasks to show
  * Slider to increase "Event Cover" priority weight
* Table view + compact bar chart

---

## 🧑‍🤝‍🧑 Team Comparison

* Compare task load and status per team member.
* Toggle between `Designer Name` and `Assigned By`.
* Bar chart with grouped bars for Completed and Remaining.
* Inline top-N selector to limit view.
* Tables showing top performers in each category.

---

## 📂 Tasks by Content Type

* Horizontal bar chart displaying task volume by `Content Type`.

---

## 📅 Weekday Completion Distribution

* Shows number of tasks completed per weekday.
* Dynamically reflects global filters and date range.

---

## 📋 Completed & Remaining Tasks Tables

* **Inline Content Type filter** for each table
* Columns: Title, Designer, Assigned By, Assigned Date, Completion Date, Design Status, and Content Type (when present)
* Fully responsive to all global filters

---

## 🕒 Remaining Tasks with Posting Info

* Shows only tasks with `Status = Remaining`
* Columns: Content Type, Title, Deadline, Assigned By, and interpreted Posting Status

  * ✅ Posted
  * ❌ Not Posted
  * 💬 (comment)
* Sorting options:

  * Prioritize unposted tasks
  * Sort by Days Remaining until Deadline

---

## 🧾 CSV Export

* Download button to export the current filtered DataFrame.

---

## 🚀 Setup Notes

* Built with **Streamlit** and **Plotly**
* Uses **Google Sheets API** for live data integration
* Filters, charts, and tables are dynamically reactive
* Supports light and dark themes automatically


## 📁 Repository Structure

```
📦 FES-Pulse(main files)
├── 📄 app.py               # Main Streamlit app logic
├── 📄 README.md            # You're here!
├── 📄 theme.css            #dark mode n light mode css
s
```

---

## 🙌 Credits

Crafted with ❤️ by Sameer Raza Malik using Python, Streamlit, Pandas, and Plotly.

---

## 📬 Feedback & Issues

Have suggestions or found a bug? Open an issue or drop a message.

---

*"FES Pulse — Tracking every beat of your team's productivity."*
