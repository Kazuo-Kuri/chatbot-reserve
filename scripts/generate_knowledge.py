import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os

# .env読み込み（ローカル実行時）
if os.getenv("GITHUB_ACTIONS") != "true":
    load_dotenv()

# 認証設定
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
gc = gspread.authorize(credentials)

# スプレッドシートとシート名の取得
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID") or "1ApH-A58jUCZSKwTBAyuPZlZTNsv_2RwKGSqZNyaHHfk"
KNOWLEDGE_SHEET = os.getenv("KNOWLEDGE_SHEET") or "knowledge"  # ✅ 共通データとして "knowledge"

# スプレッドシート読み込み
spreadsheet = gc.open_by_key(SPREADSHEET_ID)
sheet = spreadsheet.worksheet(KNOWLEDGE_SHEET)

# データ取得
records = sheet.get_all_records()
knowledge = {row['title']: [row['content']] for row in records}

# 保存先固定（共通用）
os.makedirs("data", exist_ok=True)
with open("data/knowledge.json", "w", encoding="utf-8") as f:
    json.dump(knowledge, f, ensure_ascii=False, indent=2)

print("✅ data/knowledge.json を保存しました。")
