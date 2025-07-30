import re
import gspread
from google.oauth2.service_account import Credentials

SHEET_URL = "https://docs.google.com/spreadsheets/d/1yaW7V7hSBqOBZYbqIUKsGrhB8pVtWWrzq7t5scq3JVI/edit?gid=0#gid=0"

# Minimum read-only scopes; include Drive to open by URL.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

creds = Credentials.from_service_account_file("fes-employee-eda-01c012142a64.json", scopes=SCOPES)
gc = gspread.authorize(creds)

# Safer: open by key instead of URL (avoids Drive listing issues).
m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", SHEET_URL)
sheet_id = m.group(1) if m else None
sh = gc.open_by_key(sheet_id) if sheet_id else gc.open_by_url(SHEET_URL)

print("OK: opened spreadsheet:", sh.title)
print("Worksheets:", [ws.title for ws in sh.worksheets()])
ws = sh.sheet1
print("First row:", ws.row_values(1))
