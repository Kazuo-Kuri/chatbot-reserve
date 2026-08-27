import json
import os

import faiss
import numpy as np
import openai

if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set or empty.")

FAQ_PATH = "data/reserve_faq.json"
KNOWLEDGE_PATH = "data/reserve_knowledge.json"
METADATA_PATH = "data/reserve_metadata.json"
VECTOR_PATH = "data/reserve_vector_data.npy"
INDEX_PATH = "data/reserve_index.faiss"
EMBED_MODEL = "text-embedding-3-small"


def get_embeddings_in_batches(texts, batch_size=100):
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = openai.embeddings.create(
            model=EMBED_MODEL,
            input=texts[i:i + batch_size],
        )
        vectors.extend(
            np.array(item.embedding, dtype="float32")
            for item in response.data
        )
    return np.array(vectors, dtype="float32")


with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_items = json.load(f)

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    knowledge_dict = json.load(f)

if not isinstance(faq_items, list):
    raise ValueError("reserve_faq.json must contain a list.")
if not isinstance(knowledge_dict, dict):
    raise ValueError("reserve_knowledge.json must contain an object.")

faq_contents = [
    f"{item['question']} {item['answer']}"
    for item in faq_items
    if item.get("question") and item.get("answer")
]
knowledge_contents = [
    f"{category}：{text}"
    for category, texts in knowledge_dict.items()
    for text in texts
]

search_corpus = faq_contents + knowledge_contents
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata_note = (
        f"【ファイル情報】{metadata.get('title', '')}"
        f"（種類：{metadata.get('type', '')}、優先度：{metadata.get('priority', '')}）"
    )
    if metadata_note.strip():
        search_corpus.append(metadata_note)

search_corpus = [text for text in search_corpus if text.strip()]
if not search_corpus:
    raise ValueError("No reserve FAQ or Knowledge content is available to index.")

print("🔄 reserve FAQとKnowledgeのベクトルをバッチで再生成しています...")
vector_data = get_embeddings_in_batches(search_corpus)

index = faiss.IndexFlatL2(vector_data.shape[1])
index.add(vector_data)

np.save(VECTOR_PATH, vector_data)
faiss.write_index(index, INDEX_PATH)

print("✅ 予約用ベクトルデータとFAISSインデックスを保存しました。")
