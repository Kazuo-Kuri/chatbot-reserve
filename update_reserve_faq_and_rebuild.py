import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
with open("credentials.json", "r", encoding="utf-8") as f:
    credentials_info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(
    credentials_info, scopes=SCOPES
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
RANGE_NAME = os.getenv("FAQ_RANGE", "reserve_faq!A1:C")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/reserve_faq.json")

sheet_service = build('sheets', 'v4', credentials=credentials).spreadsheets()
result = sheet_service.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
values = result.get('values', [])

faq_list = []
for row in values[1:]:
    if len(row) >= 2 and row[0].strip() and row[1].strip():
        entry = {"question": row[0].strip(), "answer": row[1].strip()}
        if len(row) >= 3 and row[2].strip():
            entry["category"] = row[2].strip()
        faq_list.append(entry)

os.makedirs("data", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=2)

print(f"✅ {OUTPUT_PATH} を保存しました。")
