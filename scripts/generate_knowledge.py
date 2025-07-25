import gspread
import json
import time
import numpy as np
import openai
import faiss
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os

# .env読み込み（ローカル実行時）
if os.getenv("GITHUB_ACTIONS") != "true":
    load_dotenv()

# OpenAI APIキー設定
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

# 認証設定
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
gc = gspread.authorize(credentials)

# スプレッドシートとシート名の取得
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID") or "1ApH-A58jUCZSKwTBAyuPZlZTNsv_2RwKGSqZNyaHHfk"
KNOWLEDGE_SHEET = os.getenv("KNOWLEDGE_SHEET") or "reserve_knowledge"  # ← 環境変数が優先

# スプレッドシート読み込み
spreadsheet = gc.open_by_key(SPREADSHEET_ID)
sheet = spreadsheet.worksheet(KNOWLEDGE_SHEET)

# データ取得
records = sheet.get_all_records()
knowledge = {row['title']: [row['content']] for row in records}

# 保存
os.makedirs("data", exist_ok=True)
with open("data/knowledge.json", "w", encoding="utf-8") as f:
    json.dump(knowledge, f, ensure_ascii=False, indent=2)

print("✅ data/knowledge.json を保存しました。")

# ベクトル埋め込み処理
texts = [f"{title}：{content[0]}" for title, content in knowledge.items()]
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

def get_embeddings_batch(text_batch, retries=3, delay=3):
    for attempt in range(retries):
        try:
            response = openai.embeddings.create(model=EMBED_MODEL, input=text_batch)
            return [np.array(d.embedding, dtype="float32") for d in response.data]
        except openai.error.OpenAIError as e:
            print(f"⚠️ API error: {e}. Retrying in {delay} sec...")
            time.sleep(delay)
    raise RuntimeError("❌ Failed to get embeddings after multiple retries.")

print("🔄 ベクトルを再生成しています...")

all_vectors = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    vectors = get_embeddings_batch(batch)
    all_vectors.extend(vectors)
    print(f"✅ Processed {i + len(batch)}/{len(texts)}")

vector_data = np.array(all_vectors, dtype="float32")

# FAISSインデックス作成・保存
dimension = vector_data.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vector_data)

np.save("data/vector_data.npy", vector_data)
faiss.write_index(index, "data/index.faiss")

print("✅ ベクトルデータとFAISSインデックスを保存しました。")
