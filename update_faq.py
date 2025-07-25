import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# === 環境変数から読み込み（GitHub Actions対応）===
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID") or '1ApH-A58jUCZSKwTBAyuPZlZTNsv_2RwKGSqZNyaHHfk'
RANGE_NAME = 'reserve_faq!A1:C'  # ← reserve用に変更済み

# === credentials.json を読み込み ===
with open("credentials.json", "r", encoding="utf-8") as f:
    credentials_info = json.load(f)

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
credentials = service_account.Credentials.from_service_account_info(
    credentials_info, scopes=SCOPES
)

# === Sheets API クライアントを作成 ===
sheet = build('sheets', 'v4', credentials=credentials).spreadsheets()

# === データ取得 ===
result = sheet.values().get(
    spreadsheetId=SPREADSHEET_ID,
    range=RANGE_NAME
).execute()
values = result.get('values', [])

# === JSON化処理 ===
faq_list = []
for row in values[1:]:  # 1行目はヘッダー
    if len(row) >= 2 and row[0].strip() and row[1].strip():
        faq = {
            'question': row[0].strip(),
            'answer': row[1].strip()
        }
        if len(row) >= 3 and row[2].strip():
            faq['category'] = row[2].strip()
        faq_list.append(faq)

# === JSONファイルとして出力 ===
os.makedirs('data', exist_ok=True)
with open('data/faq.json', 'w', encoding='utf-8') as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=2)

print("✅ data/faq.json を生成しました。")
