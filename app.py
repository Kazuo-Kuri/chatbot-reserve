# app.py
import os
import json
import time
import base64
import ipaddress
import threading
import traceback
from datetime import datetime
from typing import Any, cast

from dotenv import load_dotenv

# proxy 環境変数の削除
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS
from flask_limiter import Limiter
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
import faiss
import numpy as np

from product_film_matcher import ProductFilmMatcher
from query_expander import expand_query
from expand_reserve_query import expand_reserve_query


# ============================================================
# 1. 共通設定
# ============================================================

EMBED_MODEL = "text-embedding-3-small"

VECTOR_PATH = "data/vector_data.npy"
INDEX_PATH = "data/index.faiss"

RESERVE_VECTOR_PATH = "data/reserve_vector_data.npy"
RESERVE_INDEX_PATH = "data/reserve_index.faiss"

load_dotenv()


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

DEFAULT_ALLOWED_ORIGIN = "https://chatbot-re.psi-coffee.com"

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGIN).split(",")
    if origin.strip()
]

if not allowed_origins:
    allowed_origins = [DEFAULT_ALLOWED_ORIGIN]

CORS(
    app,
    resources={
        r"/chat": {"origins": allowed_origins},
        r"/feedback": {"origins": allowed_origins},
    },
)


# ============================================================
# Rate Limit
# ============================================================

def get_rate_limit_key():
    # Renderの公開WebサービスではCloudflareがCF-Connecting-IPを上書きする。
    # 呼び出し元が左端を偽装できるX-Forwarded-Forはレート制限に使用しない。
    if os.getenv("RENDER", "").lower() == "true":
        client_ip = request.headers.get("CF-Connecting-IP", "").strip()
        try:
            return str(ipaddress.ip_address(client_ip))
        except ValueError:
            pass

    return request.remote_addr or "unknown"


limiter = Limiter(
    key_func=get_rate_limit_key,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


# ============================================================
# OpenAI
# ============================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# セッション履歴
# ============================================================

session_histories: dict[str, dict[str, Any]] = {}
HISTORY_TTL = 1800


def get_session_history(session_id: str):
    now = time.time()
    session = session_histories.get(session_id)

    if not session or now - session["last_active"] > HISTORY_TTL:
        session_histories[session_id] = {
            "last_active": now,
            "history": [],
        }
    else:
        session_histories[session_id]["last_active"] = now

    return session_histories[session_id]["history"]


def add_to_session_history(session_id: str, role: str, content: str):
    history = get_session_history(session_id)
    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    if len(history) > 10:
        history[:] = history[-10:]


# ============================================================
# Embedding
# ============================================================

def get_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("空のテキストには埋め込みを生成できません")

    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[text],
        )

        if not response.data or not response.data[0].embedding:
            raise ValueError("埋め込みデータが空です")

        return np.array(
            response.data[0].embedding,
            dtype="float32",
        )

    except Exception as e:
        print("❌ Embedding error:", e)
        raise


# ============================================================
# 通常用 FAQ
# ============================================================

with open("data/faq.json", encoding="utf-8") as f:
    faq_items = json.load(f)

faq_questions = [
    item["question"]
    for item in faq_items
]

faq_answers = [
    item["answer"]
    for item in faq_items
]


# ============================================================
# 通常用 Knowledge
# ============================================================

with open("data/knowledge.json", encoding="utf-8") as f:
    knowledge_dict = json.load(f)

knowledge_contents = [
    f"{category}：{text}"
    for category, texts in knowledge_dict.items()
    for text in texts
]


# ============================================================
# 通常用 Metadata
# ============================================================

metadata_note = ""
metadata: dict[str, Any] = {}
metadata_path = "data/metadata.json"

if os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as f:
        loaded_metadata = json.load(f)

    if isinstance(loaded_metadata, dict):
        metadata = loaded_metadata

    metadata_note = (
        f"{metadata.get('title', '')} "
        f"(種類: {metadata.get('type', '')}, "
        f"優先度: {metadata.get('priority', '')})"
    )


# 通常用生成スクリプトは末尾にメタ情報を1件追加する。
index_metadata_note = ""

if metadata_note:
    index_metadata_note = (
        f"【ファイル情報】"
        f"{metadata.get('title', '')}"
        f"（種類：{metadata.get('type', '')}、"
        f"優先度：{metadata.get('priority', '')}）"
    )

search_corpus = (
    faq_questions
    + knowledge_contents
    + [index_metadata_note]
)

source_flags = (
    ["faq"] * len(faq_questions)
    + ["knowledge"] * len(knowledge_contents)
    + ["metadata"]
)


# ============================================================
# 予約システム用 FAQ
# ============================================================

with open("data/reserve_faq.json", encoding="utf-8") as f:
    reserve_faq_items = json.load(f)

reserve_faq_items = [
    item
    for item in reserve_faq_items
    if item.get("question") and item.get("answer")
]

reserve_faq_questions = [
    item["question"]
    for item in reserve_faq_items
]

reserve_faq_answers = [
    item["answer"]
    for item in reserve_faq_items
]


# ============================================================
# 予約システム用 Knowledge
# ============================================================

with open("data/reserve_knowledge.json", encoding="utf-8") as f:
    reserve_knowledge_dict = json.load(f)

reserve_knowledge_contents = [
    f"{category}：{text}"
    for category, texts in reserve_knowledge_dict.items()
    for text in texts
]


# ============================================================
# 予約システム用 検索対象とフラグ
# ============================================================

reserve_search_corpus = [
    f"{q} {a}"
    for q, a in zip(
        reserve_faq_questions,
        reserve_faq_answers,
    )
] + reserve_knowledge_contents

reserve_source_flags = (
    ["faq"] * len(reserve_faq_questions)
    + ["knowledge"] * len(reserve_knowledge_contents)
)


# ============================================================
# 予約システム用 Metadata
# ============================================================

reserve_metadata_path = "data/reserve_metadata.json"

if os.path.exists(reserve_metadata_path):
    with open(reserve_metadata_path, encoding="utf-8") as f:
        loaded_reserve_metadata = json.load(f)

    reserve_metadata: dict[str, Any] = {}

    if isinstance(loaded_reserve_metadata, dict):
        reserve_metadata = loaded_reserve_metadata

    reserve_search_corpus.append(
        f"【ファイル情報】"
        f"{reserve_metadata.get('title', '')}"
        f"（種類：{reserve_metadata.get('type', '')}、"
        f"優先度：{reserve_metadata.get('priority', '')}）"
    )

    reserve_source_flags.append("metadata")


# ============================================================
# FAISS
#
# faiss のPython型定義がPylanceに完全対応していないため、
# cast(Any, faiss) を通して型チェック上の誤警告を回避する。
# 実行処理自体は従来と同じ。
# ============================================================

faiss_api = cast(Any, faiss)


# ============================================================
# 通常用 FAISS
# ============================================================

if os.path.exists(VECTOR_PATH) and os.path.exists(INDEX_PATH):
    vector_data = np.load(VECTOR_PATH)
    index = faiss_api.read_index(INDEX_PATH)

else:
    vector_data = np.array(
        [
            get_embedding(text)
            for text in search_corpus
        ],
        dtype="float32",
    )

    index = faiss_api.IndexFlatL2(
        vector_data.shape[1]
    )

    index.add(vector_data)

    np.save(
        VECTOR_PATH,
        vector_data,
    )

    faiss_api.write_index(
        index,
        INDEX_PATH,
    )


# ============================================================
# 予約システム用 FAISS
# ============================================================

if (
    os.path.exists(RESERVE_VECTOR_PATH)
    and os.path.exists(RESERVE_INDEX_PATH)
):
    reserve_vector_data = np.load(
        RESERVE_VECTOR_PATH
    )

    reserve_index = faiss_api.read_index(
        RESERVE_INDEX_PATH
    )

else:
    reserve_vector_data = np.array(
        [
            get_embedding(text)
            for text in reserve_search_corpus
        ],
        dtype="float32",
    )

    reserve_index = faiss_api.IndexFlatL2(
        reserve_vector_data.shape[1]
    )

    reserve_index.add(
        reserve_vector_data
    )

    np.save(
        RESERVE_VECTOR_PATH,
        reserve_vector_data,
    )

    faiss_api.write_index(
        reserve_index,
        RESERVE_INDEX_PATH,
    )


# ============================================================
# チャットボット種別判定
#
# 質問内容ではなく、アクセス元Originで
# 予約システム用チャットボットかどうかを判定する。
# ============================================================

RESERVE_CHAT_ORIGIN = os.getenv(
    "RESERVE_CHAT_ORIGIN",
    "https://chatbot-re.psi-coffee.com",
).rstrip("/")


def is_reserve_chat_request():
    origin = request.headers.get(
        "Origin",
        "",
    ).rstrip("/")

    return origin == RESERVE_CHAT_ORIGIN


# ============================================================
# インデックス整合性チェック
# ============================================================

def validate_search_index(
    vectors,
    search_index,
    corpus,
    label,
):
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(corpus)
        or search_index.ntotal != len(corpus)
        or vectors.shape[1] != search_index.d
    ):
        raise ValueError(
            f"{label}: "
            "corpus / vectors / FAISS の"
            "件数または次元が不一致です"
        )


validate_search_index(
    vector_data,
    index,
    search_corpus,
    "通常",
)

validate_search_index(
    reserve_vector_data,
    reserve_index,
    reserve_search_corpus,
    "予約",
)


# ============================================================
# Google Sheets
# ============================================================

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

UNANSWERED_SHEET = "faq_suggestions_reserve"
FEEDBACK_SHEET = "feedback_log_reserve"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets"
]

credentials_base64 = (
    os.getenv("GOOGLE_CREDENTIALS")
    or os.getenv("GOOGLE_CREDENTIALS_BASE64")
)

if not credentials_base64:
    raise ValueError(
        "GOOGLE_CREDENTIALS または "
        "GOOGLE_CREDENTIALS_BASE64 が設定されていません。"
    )

credentials_info = json.loads(
    base64.b64decode(
        credentials_base64
    ).decode("utf-8")
)

credentials = (
    service_account.Credentials
    .from_service_account_info(
        credentials_info,
        scopes=SCOPES,
    )
)

sheet_service = build(
    "sheets",
    "v4",
    credentials=credentials,
).spreadsheets()

sheet_id_cache: dict[str, int] = {}
sheet_write_lock = threading.Lock()


# ============================================================
# Product Film Matcher
# ============================================================

pf_matcher = ProductFilmMatcher(
    "data/product_film_color_matrix.json"
)


# ============================================================
# System Prompt
# ============================================================

with open("system_prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()


# ============================================================
# 回答モード
# ============================================================

def infer_response_mode(question: str):
    q_len = len(question)

    if q_len < 30:
        return "short"

    if q_len > 100:
        return "long"

    return "default"


# ============================================================
# Chat Log
# ============================================================

CHAT_LOG_SHEET = "chat_logs_reserve"


# ============================================================
# Google Sheets 行追加
# ============================================================

def insert_log_row(
    sheet_name,
    values,
):
    with sheet_write_lock:
        sheet_id = sheet_id_cache.get(
            sheet_name
        )

        if sheet_id is None:
            spreadsheet = (
                sheet_service.get(
                    spreadsheetId=SPREADSHEET_ID,
                    fields=(
                        "sheets.properties"
                        "(sheetId,title)"
                    ),
                ).execute()
            )

            sheet_id = next(
                (
                    sheet["properties"]["sheetId"]
                    for sheet in spreadsheet.get(
                        "sheets",
                        [],
                    )
                    if (
                        sheet.get(
                            "properties",
                            {},
                        ).get("title")
                        == sheet_name
                    )
                ),
                None,
            )

            if sheet_id is None:
                raise ValueError(
                    f"Sheet not found: {sheet_name}"
                )

            sheet_id_cache[
                sheet_name
            ] = sheet_id

        cell_values = []

        for value in values:
            if isinstance(
                value,
                bool,
            ):
                user_entered_value = {
                    "boolValue": value
                }

            elif isinstance(
                value,
                (int, float),
            ):
                user_entered_value = {
                    "numberValue": value
                }

            else:
                user_entered_value = {
                    "stringValue": str(value)
                }

            cell_values.append(
                {
                    "userEnteredValue":
                    user_entered_value
                }
            )

        sheet_service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={
                "requests": [
                    {
                        "insertDimension": {
                            "range": {
                                "sheetId": sheet_id,
                                "dimension": "ROWS",
                                "startIndex": 1,
                                "endIndex": 2,
                            },
                            "inheritFromBefore": False,
                        }
                    },
                    {
                        "updateCells": {
                            "start": {
                                "sheetId": sheet_id,
                                "rowIndex": 1,
                                "columnIndex": 0,
                            },
                            "rows": [
                                {
                                    "values": cell_values
                                }
                            ],
                            "fields": "userEnteredValue",
                        }
                    },
                ]
            },
        ).execute()


# ============================================================
# ログ記録
# ============================================================

def log_chat_history(
    user_q,
    answer,
    source_type,
    is_unanswered,
):
    try:
        insert_log_row(
            CHAT_LOG_SHEET,
            [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                user_q.strip(),
                answer.strip(),
                source_type,
                str(
                    is_unanswered
                ).lower(),
            ],
        )

    except Exception as e:
        print(
            "❌ ログ出力失敗:",
            e,
        )


# ============================================================
# Greeting
# ============================================================

GREETING_PATTERNS = [
    "こんにちは",
    "こんばんは",
    "おはよう",
    "はじめまして",
    "宜しくお願いします",
    "よろしくお願いします",
]


# ============================================================
# JSON Request
# ============================================================

def parse_json_request():
    if not request.is_json:
        return (
            None,
            (
                jsonify(
                    {
                        "error":
                        "JSON形式で送信してください。"
                    }
                ),
                415,
            ),
        )

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict,
    ):
        return (
            None,
            (
                jsonify(
                    {
                        "error":
                        "正しいJSON形式で送信してください。"
                    }
                ),
                400,
            ),
        )

    return data, None


# ============================================================
# Session ID
# ============================================================

def is_valid_session_id(
    session_id,
):
    return (
        isinstance(
            session_id,
            str,
        )
        and bool(
            session_id.strip()
        )
        and len(
            session_id
        ) <= 128
        and all(
            char.isprintable()
            for char in session_id
        )
    )


# ============================================================
# Error Handler
# ============================================================

@app.errorhandler(413)
def request_too_large(
    _error,
):
    return jsonify(
        {
            "error":
            (
                "送信データが大きすぎます。"
                "内容を短くして"
                "再度お試しください。"
            )
        }
    ), 413


@app.errorhandler(429)
def rate_limit_exceeded(
    _error,
):
    return jsonify(
        {
            "error":
            (
                "リクエストが多すぎます。"
                "時間をおいて"
                "再度お試しください。"
            )
        }
    ), 429


# ============================================================
# Chat API
# ============================================================

@app.route(
    "/chat",
    methods=["POST"],
)
@limiter.limit(
    "10 per minute; 100 per hour"
)
def chat():
    try:
        data, error_response = (
            parse_json_request()
        )

        if error_response is not None:
            return error_response

        if data is None:
            return jsonify(
                {
                    "error":
                    "正しいJSON形式で送信してください。"
                }
            ), 400

        question = data.get(
            "question",
            "",
        )

        session_id = data.get(
            "session_id",
            "",
        )

        if not isinstance(
            question,
            str,
        ):
            return jsonify(
                {
                    "error":
                    "質問は文字列で入力してください。"
                }
            ), 400

        user_q = question.strip()

        if not user_q:
            return jsonify(
                {
                    "error":
                    "質問がありません"
                }
            ), 400

        if len(user_q) > 2000:
            return jsonify(
                {
                    "error":
                    "質問は2000文字以内で入力してください。"
                }
            ), 400

        if not is_valid_session_id(
            session_id
        ):
            return jsonify(
                {
                    "error":
                    "有効なセッションIDを指定してください。"
                }
            ), 400

        session_id = cast(
            str,
            session_id,
        )

        # ----------------------------------------------------
        # Greeting
        # ----------------------------------------------------

        if any(
            greet in user_q
            for greet in GREETING_PATTERNS
        ):
            reply = (
                "こんにちは！"
                "ご質問があれば"
                "お気軽にどうぞ。"
            )

            add_to_session_history(
                session_id,
                "assistant",
                reply,
            )

            return jsonify(
                {
                    "response": reply,
                    "original_question": user_q,
                    "expanded_question": user_q,
                }
            )

        # ----------------------------------------------------
        # Session History
        # ----------------------------------------------------

        session_history = list(
            get_session_history(
                session_id
            )
        )

        add_to_session_history(
            session_id,
            "user",
            user_q,
        )

        # ----------------------------------------------------
        # Chatbot Type
        #
        # 質問文ではなく、アクセス元Originで判定
        # ----------------------------------------------------

        use_reserve = (
            is_reserve_chat_request()
        )

        # ----------------------------------------------------
        # Query Expansion
        # ----------------------------------------------------

        if use_reserve:
            expanded_q = (
                expand_reserve_query(
                    user_q,
                    session_history,
                )
            )

        else:
            expanded_q = (
                expand_query(
                    user_q,
                    session_history,
                )
            )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        q_vector = get_embedding(
            expanded_q
        )

        query_vector = np.array(
            [q_vector],
            dtype="float32",
        )

        # ----------------------------------------------------
        # FAISS Search
        #
        # 通常チャット:
        #   通常FAQ / Knowledgeのみ
        #
        # 予約チャット:
        #   通常FAQ / Knowledge
        #   + reserve FAQ / Knowledge
        # ----------------------------------------------------

        search_results = []

        def add_search_results(
            search_index,
            search_flags,
            search_faq_questions,
            search_faq_answers,
            search_knowledge_contents,
            source_prefix="",
        ):
            search_index_api = cast(
                Any,
                search_index,
            )

            distances, indexes = (
                search_index_api.search(
                    query_vector,
                    7,
                )
            )

            for distance, idx in zip(
                distances[0],
                indexes[0],
            ):
                idx = int(idx)

                if (
                    idx < 0
                    or idx >= len(
                        search_flags
                    )
                ):
                    continue

                src = search_flags[
                    idx
                ]

                if src == "faq":
                    if idx >= len(
                        search_faq_questions
                    ):
                        continue

                    search_results.append(
                        {
                            "distance":
                            float(distance),
                            "type":
                            f"{source_prefix}faq",
                            "question":
                            search_faq_questions[
                                idx
                            ],
                            "answer":
                            search_faq_answers[
                                idx
                            ],
                        }
                    )

                elif src == "knowledge":
                    ref_idx = (
                        idx
                        - len(
                            search_faq_questions
                        )
                    )

                    if (
                        0 <= ref_idx
                        < len(
                            search_knowledge_contents
                        )
                    ):
                        search_results.append(
                            {
                                "distance":
                                float(distance),
                                "type":
                                (
                                    f"{source_prefix}"
                                    "knowledge"
                                ),
                                "content":
                                search_knowledge_contents[
                                    ref_idx
                                ],
                            }
                        )

        # ----------------------------------------------------
        # 通常FAQ / Knowledge
        #
        # 通常・予約の両チャットで必ず検索
        # ----------------------------------------------------

        add_search_results(
            index,
            source_flags,
            faq_questions,
            faq_answers,
            knowledge_contents,
        )

        # ----------------------------------------------------
        # reserve FAQ / Knowledge
        #
        # 予約システム用チャットだけ追加検索
        # ----------------------------------------------------

        if use_reserve:
            add_search_results(
                reserve_index,
                reserve_source_flags,
                reserve_faq_questions,
                reserve_faq_answers,
                reserve_knowledge_contents,
                "reserve_",
            )

        # ----------------------------------------------------
        # 距離が近い順に統合
        # ----------------------------------------------------

        search_results.sort(
            key=lambda item:
            item["distance"]
        )

        faq_context = []
        reference_context = []
        used_sources = set()

        for result in search_results:
            result_type = result[
                "type"
            ]

            if (
                result_type
                in [
                    "faq",
                    "reserve_faq",
                ]
            ):
                # FAQは最大3件
                if len(
                    faq_context
                ) >= 3:
                    continue

                faq_context.append(
                    (
                        f"Q: "
                        f"{result['question']}"
                        f"\nA: "
                        f"{result['answer']}"
                    )
                )

                used_sources.add(
                    result_type
                )

            elif (
                result_type
                in [
                    "knowledge",
                    "reserve_knowledge",
                ]
            ):
                # Knowledgeは最大2件
                if len(
                    reference_context
                ) >= 2:
                    continue

                reference_context.append(
                    (
                        "【参考知識】"
                        f"{result['content']}"
                    )
                )

                used_sources.add(
                    result_type
                )

        # ----------------------------------------------------
        # Film Matcher
        #
        # 従来どおり通常チャットのみ
        # ----------------------------------------------------

        film_info_text = ""

        if not use_reserve:
            film_match_data = (
                pf_matcher.match(
                    user_q,
                    session_history,
                )
            )

            film_info_text = (
                pf_matcher
                .format_match_info(
                    film_match_data
                )
            )

        if film_info_text:
            reference_context.insert(
                0,
                film_info_text,
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if metadata_note:
            reference_context.append(
                (
                    "【参考ファイル情報】"
                    f"{metadata_note}"
                )
            )

        # ----------------------------------------------------
        # No Context
        # ----------------------------------------------------

        if (
            not faq_context
            and not reference_context
            and not film_info_text.strip()
        ):
            answer = (
                "当社はコーヒー製品の"
                "委託加工を専門とする会社です。"
                "恐れ入りますが、"
                "ご質問内容が当社業務と"
                "直接関連のある内容かどうかを"
                "ご確認のうえ、"
                "改めてお尋ねいただけますと"
                "幸いです。\n\n"
                "ご不明な点がございましたら、"
                "当社の【お問い合わせフォーム】"
                "よりご連絡ください。"
            )

            add_to_session_history(
                session_id,
                "assistant",
                answer,
            )

            return jsonify(
                {
                    "response":
                    answer,
                    "original_question":
                    user_q,
                    "expanded_question":
                    expanded_q,
                }
            )

        # ----------------------------------------------------
        # Prompt Context
        # ----------------------------------------------------

        faq_part = (
            "\n\n".join(
                faq_context[:3]
            )
            if faq_context
            else
            "該当するFAQは見つかりませんでした。"
        )

        ref_texts = [
            text
            for text in reference_context
            if (
                "製品フィルム・カラー情報"
                in text
            )
        ]

        other_refs = [
            text
            for text in reference_context
            if (
                "製品フィルム・カラー情報"
                not in text
            )
        ][:2]

        ref_part = "\n".join(
            ref_texts
            + other_refs
        )

        mode = infer_response_mode(
            user_q
        )

        prompt = f"""以下は当社のFAQおよび参考情報です。これらを参考に、ユーザーの質問に製造元の立場でご回答ください。

【FAQ】
{faq_part}

【参考情報】
{ref_part}

ユーザーの質問: {user_q}
回答："""

        # ----------------------------------------------------
        # System Prompt
        # ----------------------------------------------------

        system_prompt = base_prompt

        if mode == "short":
            system_prompt += (
                "\n\n"
                "可能な限り簡潔かつ"
                "要点のみで回答してください。"
            )

        elif mode == "long":
            system_prompt += (
                "\n\n"
                "詳細な説明や具体例を含めて"
                "丁寧に回答してください。"
            )

        # ----------------------------------------------------
        # GPT
        # ----------------------------------------------------

        completion = (
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role":
                        "system",
                        "content":
                        system_prompt,
                    },
                    {
                        "role":
                        "user",
                        "content":
                        prompt,
                    },
                ],
                temperature=0.2,
            )
        )

        answer = (
            completion
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not answer:
            answer = (
                "回答を生成できませんでした。"
                "お手数ですが、"
                "時間をおいて再度"
                "お試しください。"
            )

        # ----------------------------------------------------
        # 未回答ログ
        # ----------------------------------------------------

        if (
            "申し訳" in answer
            or "恐れ入りますが" in answer
            or "エラー" in answer
        ):
            try:
                insert_log_row(
                    UNANSWERED_SHEET,
                    [
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        user_q,
                        "未回答",
                        1,
                    ],
                )

            except Exception:
                print(
                    "[ERROR writing unanswered log]"
                )
                traceback.print_exc()

        # ----------------------------------------------------
        # Session History
        # ----------------------------------------------------

        add_to_session_history(
            session_id,
            "assistant",
            answer,
        )

        # ----------------------------------------------------
        # 回答ソース
        # ----------------------------------------------------

        if used_sources:
            source_type = "+".join(
                sorted(
                    used_sources
                )
            )
        else:
            source_type = (
                "reserve"
                if use_reserve
                else "normal"
            )

        is_unanswered = any(
            phrase in answer
            for phrase in [
                "申し訳",
                "恐れ入りますが",
                "エラー",
            ]
        )

        log_chat_history(
            user_q,
            answer,
            source_type,
            is_unanswered,
        )

        return jsonify(
            {
                "response":
                answer,
                "original_question":
                user_q,
                "expanded_question":
                expanded_q,
            }
        )

    except RequestEntityTooLarge:
        raise

    except Exception:
        print(
            "[ERROR in /chat]"
        )

        traceback.print_exc()

        return jsonify(
            {
                "response":
                (
                    "一時的なエラーが発生しました。"
                    "時間をおいて"
                    "再度お試しください。"
                )
            }
        ), 500


# ============================================================
# Feedback API
# ============================================================

@app.route(
    "/feedback",
    methods=["POST"],
)
@limiter.limit(
    "20 per minute; 200 per hour"
)
def feedback():
    data, error_response = (
        parse_json_request()
    )

    if error_response is not None:
        return error_response

    if data is None:
        return jsonify(
            {
                "error":
                "正しいJSON形式で送信してください。"
            }
        ), 400

    question = data.get(
        "question",
        "",
    )

    answer = data.get(
        "answer",
        "",
    )

    feedback_value = data.get(
        "feedback",
        "",
    )

    reason = data.get(
        "reason",
        "",
    )

    if not isinstance(
        question,
        str,
    ):
        return jsonify(
            {
                "error":
                "フィードバック項目は文字列で送信してください。"
            }
        ), 400

    if not isinstance(
        answer,
        str,
    ):
        return jsonify(
            {
                "error":
                "フィードバック項目は文字列で送信してください。"
            }
        ), 400

    if not isinstance(
        feedback_value,
        str,
    ):
        return jsonify(
            {
                "error":
                "フィードバック項目は文字列で送信してください。"
            }
        ), 400

    if not isinstance(
        reason,
        str,
    ):
        return jsonify(
            {
                "error":
                "フィードバック項目は文字列で送信してください。"
            }
        ), 400

    question = question.strip()
    answer = answer.strip()
    feedback_value = feedback_value.strip()
    reason = reason.strip()

    if not all(
        [
            question,
            answer,
            feedback_value,
        ]
    ):
        return jsonify(
            {
                "error":
                "不完全なフィードバックデータです"
            }
        ), 400

    if (
        len(question) > 2000
        or len(answer) > 10000
        or len(feedback_value) > 100
        or len(reason) > 2000
    ):
        return jsonify(
            {
                "error":
                (
                    "フィードバックの入力文字数が"
                    "上限を超えています。"
                )
            }
        ), 400

    try:
        insert_log_row(
            FEEDBACK_SHEET,
            [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                question,
                answer,
                feedback_value,
                reason,
            ],
        )

    except Exception:
        print(
            "[ERROR writing feedback]"
        )
        traceback.print_exc()

        return jsonify(
            {
                "error":
                (
                    "フィードバックを"
                    "保存できませんでした。"
                    "時間をおいて"
                    "再度お試しください。"
                )
            }
        ), 503

    return jsonify(
        {
            "status":
            "success"
        }
    )


# ============================================================
# Health Check
# ============================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():
    return "Chatbot API is running."


# ============================================================
# Local Run
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
