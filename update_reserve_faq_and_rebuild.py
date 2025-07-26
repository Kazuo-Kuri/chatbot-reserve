import os
import json
import numpy as np
import faiss
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai

if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set or empty.")

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

with open("data/reserve_knowledge.json", "r", encoding="utf-8") as f:
    knowledge_dict = json.load(f)
knowledge_contents = [
    f"{category}：{text}"
    for category, texts in knowledge_dict.items()
    for text in texts
]

metadata_note = ""
metadata_path = "data/reserve_metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        metadata_note = f"【ファイル情報】{metadata.get('title', '')}（種類：{metadata.get('type', '')}、優先度：{metadata.get('priority', '')}）"

EMBED_MODEL = "text-embedding-3-small"
search_corpus = [
    f"{item['question']} {item['answer']}"
    for item in faq_list
    if item.get("question") and item.get("answer")
] + knowledge_contents + [metadata_note]

search_corpus = [s for s in search_corpus if s.strip()]

def get_embeddings_in_batches(texts, batch_size=100):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vectors = [np.array(data.embedding, dtype="float32") for data in response.data]
        vectors.extend(batch_vectors)
    return np.array(vectors, dtype="float32")

print("🔄 ベクトルをバッチで再生成しています...")
vector_data = get_embeddings_in_batches(search_corpus)

dimension = vector_data.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vector_data)

np.save("data/reserve_vector_data.npy", vector_data)
faiss.write_index(index, "data/reserve_index.faiss")

print("✅ ベクトルデータとFAISSインデックス（予約専用）を保存しました。")