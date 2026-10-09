import os
import json
from typing import Any, cast

import numpy as np
import faiss
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai


# ============================================================
# ローカル実行時のみ .env を読み込む
# ============================================================

if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()


# ============================================================
# OpenAI APIキー設定
# ============================================================

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set or empty."
    )


# ============================================================
# Google credentials
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly"
]

with open(
    "credentials.json",
    "r",
    encoding="utf-8",
) as f:
    credentials_info = json.load(f)


credentials = (
    service_account.Credentials
    .from_service_account_info(
        credentials_info,
        scopes=SCOPES,
    )
)


# ============================================================
# スプレッドシート設定
#
# 通常用FAQなので FAQ シートを読み込む
# ============================================================

SPREADSHEET_ID = (
    os.getenv("SPREADSHEET_ID")
    or "1ApH-A58jUCZSKwTBAyuPZlZTNsv_2RwKGSqZNyaHHfk"
)

RANGE_NAME = (
    os.getenv("FAQ_RANGE")
    or "FAQ!A1:C"
)


sheet_service = build(
    "sheets",
    "v4",
    credentials=credentials,
).spreadsheets()


result = (
    sheet_service
    .values()
    .get(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
    )
    .execute()
)


values = result.get(
    "values",
    [],
)


# ============================================================
# faq.json を生成
# ============================================================

faq_list = []


for row in values[1:]:
    # 1行目はヘッダー

    if (
        len(row) >= 2
        and row[0].strip()
        and row[1].strip()
    ):

        entry = {
            "question": row[0].strip(),
            "answer": row[1].strip(),
        }

        if (
            len(row) >= 3
            and row[2].strip()
        ):
            entry["category"] = (
                row[2].strip()
            )

        faq_list.append(
            entry
        )


os.makedirs(
    "data",
    exist_ok=True,
)


with open(
    "data/faq.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        faq_list,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(
    f"✅ data/faq.json を保存しました。"
    f" FAQ件数: {len(faq_list)}"
)


# ============================================================
# knowledge.json を読み込み
# ============================================================

with open(
    "data/knowledge.json",
    "r",
    encoding="utf-8",
) as f:

    knowledge_dict = json.load(f)


knowledge_contents = [
    f"{category}：{text}"
    for category, texts
    in knowledge_dict.items()
    for text in texts
]


# ============================================================
# metadata（任意）
# ============================================================

metadata_note = ""

metadata_path = (
    "data/metadata.json"
)


if os.path.exists(
    metadata_path
):

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    if isinstance(
        metadata,
        dict,
    ):

        metadata_note = (
            f"【ファイル情報】"
            f"{metadata.get('title', '')}"
            f"（種類："
            f"{metadata.get('type', '')}、"
            f"優先度："
            f"{metadata.get('priority', '')}"
            f"）"
        )


# ============================================================
# ベクトルを再生成
# ============================================================

EMBED_MODEL = (
    "text-embedding-3-small"
)


search_corpus = (
    [
        item["question"]
        for item in faq_list
    ]
    + knowledge_contents
)


if metadata_note:
    search_corpus.append(
        metadata_note
    )


if not search_corpus:
    raise ValueError(
        "FAQまたはKnowledgeのデータがありません。"
    )


def get_embeddings_in_batches(
    texts,
    batch_size=100,
):

    vectors = []

    for i in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            i:i + batch_size
        ]


        response = (
            openai.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
        )


        batch_vectors = [
            np.array(
                data.embedding,
                dtype="float32",
            )
            for data
            in response.data
        ]


        vectors.extend(
            batch_vectors
        )


    return np.array(
        vectors,
        dtype="float32",
    )


print(
    "🔄 ベクトルを"
    "バッチで再生成しています..."
)


vector_data = (
    get_embeddings_in_batches(
        search_corpus
    )
)


# ============================================================
# FAISSインデックス生成
#
# Pylanceの型定義誤警告を回避
# ============================================================

faiss_api = cast(
    Any,
    faiss,
)


dimension = (
    vector_data.shape[1]
)


index = (
    faiss_api.IndexFlatL2(
        dimension
    )
)


index.add(
    vector_data
)


# ============================================================
# 保存
# ============================================================

np.save(
    "data/vector_data.npy",
    vector_data,
)


faiss_api.write_index(
    index,
    "data/index.faiss",
)


print(
    "✅ ベクトルデータと"
    "FAISSインデックスを"
    "保存しました。"
)