import os
import json
import numpy as np
import faiss
import openai
from dotenv import load_dotenv

# === 初期設定 ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === パス設定（予約用） ===
FAQ_PATH = "data/reserve_faq.json"
KNOWLEDGE_PATH = "data/reserve_knowledge.json"
VECTOR_PATH = "data/reserve_vector_data.npy"
INDEX_PATH = "data/reserve_index.faiss"
EMBED_MODEL = "text-embedding-3-small"

# === Embedding取得関数 ===
def get_embedding(text):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# === データ読み込み ===
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_items = json.load(f)
faq_questions = [item["question"] for item in faq_items]

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

# reserve_knowledge.json は list 構造（title, content）を想定
if isinstance(knowledge_data, list):
    knowledge_contents = [
        f"{item['title']}：{item['content']}" for item in knowledge_data
    ]
else:
    raise ValueError("reserve_knowledge.json の形式が不正です。")

# === コーパス構築 ===
search_corpus = faq_questions + knowledge_contents

# === ベクトル生成 & インデックス構築 ===
print("🔄 予約用ベクトル生成中...")
vector_data = np.array([get_embedding(text) for text in search_corpus], dtype="float32")

print("🧠 FAISSインデックス作成...")
index = faiss.IndexFlatL2(vector_data.shape[1])
index.add(vector_data)

# === 保存 ===
print("💾 ファイル保存中...")
np.save(VECTOR_PATH, vector_data)
faiss.write_index(index, INDEX_PATH)

print("✅ 予約用インデックスの再構築が完了しました。")
