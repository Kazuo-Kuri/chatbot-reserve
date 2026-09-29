# app.py
import os
import json
import time
import base64
import ipaddress
import threading
import traceback
from datetime import datetime
from dotenv import load_dotenv

# 🛡️ proxy 環境変数の削除
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

# ① 共通設定（ここにパスを定義）
EMBED_MODEL = "text-embedding-3-small"
VECTOR_PATH = "data/vector_data.npy"
INDEX_PATH = "data/index.faiss"
RESERVE_VECTOR_PATH = "data/reserve_vector_data.npy"
RESERVE_INDEX_PATH = "data/reserve_index.faiss"

load_dotenv()

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

CORS(app, resources={
    r"/chat": {"origins": allowed_origins},
    r"/feedback": {"origins": allowed_origins},
})


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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

session_histories = {}
HISTORY_TTL = 1800

def get_session_history(session_id):
    now = time.time()
    session = session_histories.get(session_id)
    if not session or now - session["last_active"] > HISTORY_TTL:
        session_histories[session_id] = {"last_active": now, "history": []}
    else:
        session_histories[session_id]["last_active"] = now
    return session_histories[session_id]["history"]

def add_to_session_history(session_id, role, content):
    history = get_session_history(session_id)
    history.append({"role": role, "content": content})
    if len(history) > 10:
        history[:] = history[-10:]

def get_embedding(text):
    if not text or not text.strip():
        raise ValueError("空のテキストには埋め込みを生成できません")
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[text]
        )
        if not response.data or not response.data[0].embedding:
            raise ValueError("埋め込みデータが空です")
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print("❌ Embedding error:", e)
        raise

# 通常用
with open("data/faq.json", encoding="utf-8") as f:
    faq_items = json.load(f)
faq_questions = [item["question"] for item in faq_items]
faq_answers = [item["answer"] for item in faq_items]

with open("data/knowledge.json", encoding="utf-8") as f:
    knowledge_dict = json.load(f)
knowledge_contents = [
    f"{category}：{text}" for category, texts in knowledge_dict.items() for text in texts
]

metadata_note = ""
metadata_path = "data/metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
        metadata_note = f"{metadata.get('title', '')} (種類: {metadata.get('type', '')}, 優先度: {metadata.get('priority', '')})"

# 通常用生成スクリプトは末尾にメタ情報を1件追加する（検索後は従来通り別途追加）。
index_metadata_note = ""
if metadata_note:
    index_metadata_note = f"【ファイル情報】{metadata.get('title', '')}（種類：{metadata.get('type', '')}、優先度：{metadata.get('priority', '')}）"
search_corpus = faq_questions + knowledge_contents + [index_metadata_note]
source_flags = ["faq"] * len(faq_questions) + ["knowledge"] * len(knowledge_contents) + ["metadata"]

# ✅ 予約システム用 FAQ の読み込み
with open("data/reserve_faq.json", encoding="utf-8") as f:
    reserve_faq_items = json.load(f)
reserve_faq_items = [item for item in reserve_faq_items if item.get("question") and item.get("answer")]
reserve_faq_questions = [item["question"] for item in reserve_faq_items]
reserve_faq_answers = [item["answer"] for item in reserve_faq_items]

# ✅ 予約システム用 Knowledge の読み込み
with open("data/reserve_knowledge.json", encoding="utf-8") as f:
    reserve_knowledge_dict = json.load(f)
reserve_knowledge_contents = [
    f"{category}：{text}" for category, texts in reserve_knowledge_dict.items() for text in texts
]

# ✅ 予約システム用 検索対象とフラグ
reserve_search_corpus = [f"{q} {a}" for q, a in zip(reserve_faq_questions, reserve_faq_answers)] + reserve_knowledge_contents
reserve_source_flags = ["faq"] * len(reserve_faq_questions) + ["knowledge"] * len(reserve_knowledge_contents)

# rebuild_reserve_index.py と同じ任意の末尾メタ情報。
if os.path.exists("data/reserve_metadata.json"):
    with open("data/reserve_metadata.json", encoding="utf-8") as f:
        reserve_metadata = json.load(f)
    reserve_search_corpus.append(
        f"【ファイル情報】{reserve_metadata.get('title', '')}"
        f"（種類：{reserve_metadata.get('type', '')}、優先度：{reserve_metadata.get('priority', '')}）"
    )
    reserve_source_flags.append("metadata")

# ✅ 通常用 FAISS インデックスの読み込みまたは生成
if os.path.exists(VECTOR_PATH) and os.path.exists(INDEX_PATH):
    vector_data = np.load(VECTOR_PATH)
    index = faiss.read_index(INDEX_PATH)
else:
    vector_data = np.array([get_embedding(text) for text in search_corpus], dtype="float32")
    index = faiss.IndexFlatL2(vector_data.shape[1])
    index.add(vector_data)
    np.save(VECTOR_PATH, vector_data)
    faiss.write_index(index, INDEX_PATH)

# ✅ 予約システム用 FAISS インデックスの読み込みまたは生成
if os.path.exists(RESERVE_VECTOR_PATH) and os.path.exists(RESERVE_INDEX_PATH):
    reserve_vector_data = np.load(RESERVE_VECTOR_PATH)
    reserve_index = faiss.read_index(RESERVE_INDEX_PATH)
else:
    reserve_vector_data = np.array([get_embedding(text) for text in reserve_search_corpus], dtype="float32")
    reserve_index = faiss.IndexFlatL2(reserve_vector_data.shape[1])
    reserve_index.add(reserve_vector_data)
    np.save(RESERVE_VECTOR_PATH, reserve_vector_data)
    faiss.write_index(reserve_index, RESERVE_INDEX_PATH)

# 分類は /chat と同じキーワードを一箇所で管理する。
def is_reserve_query(user_q):
    return any(kw in user_q.lower() for kw in ["予約", "ログイン", "マニュアル", "アカウント", "登録"])


def validate_search_index(vectors, search_index, corpus, label):
    if (vectors.ndim != 2 or vectors.shape[0] != len(corpus)
            or search_index.ntotal != len(corpus)
            or vectors.shape[1] != search_index.d):
        raise ValueError(f"{label}: corpus / vectors / FAISS の件数または次元が不一致です")


validate_search_index(vector_data, index, search_corpus, "通常")
validate_search_index(reserve_vector_data, reserve_index, reserve_search_corpus, "予約")

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
UNANSWERED_SHEET = "faq_suggestions_reserve"
FEEDBACK_SHEET = "feedback_log_reserve"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

credentials_info = json.loads(base64.b64decode(os.environ["GOOGLE_CREDENTIALS"]).decode("utf-8"))
credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
sheet_service = build("sheets", "v4", credentials=credentials).spreadsheets()
sheet_id_cache = {}
sheet_write_lock = threading.Lock()

pf_matcher = ProductFilmMatcher("data/product_film_color_matrix.json")

with open("system_prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

def infer_response_mode(question):
    q_len = len(question)
    if q_len < 30:
        return "short"
    elif q_len > 100:
        return "long"
    else:
        return "default"
    
CHAT_LOG_SHEET = "chat_logs_reserve"


def insert_log_row(sheet_name, values):
    with sheet_write_lock:
        sheet_id = sheet_id_cache.get(sheet_name)
        if sheet_id is None:
            spreadsheet = sheet_service.get(
                spreadsheetId=SPREADSHEET_ID,
                fields="sheets.properties(sheetId,title)",
            ).execute()
            sheet_id = next(
                (
                    sheet["properties"]["sheetId"]
                    for sheet in spreadsheet.get("sheets", [])
                    if sheet.get("properties", {}).get("title") == sheet_name
                ),
                None,
            )
            if sheet_id is None:
                raise ValueError(f"Sheet not found: {sheet_name}")
            sheet_id_cache[sheet_name] = sheet_id

        cell_values = []
        for value in values:
            if isinstance(value, bool):
                user_entered_value = {"boolValue": value}
            elif isinstance(value, (int, float)):
                user_entered_value = {"numberValue": value}
            else:
                user_entered_value = {"stringValue": str(value)}
            cell_values.append({"userEnteredValue": user_entered_value})

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
                            "rows": [{"values": cell_values}],
                            "fields": "userEnteredValue",
                        }
                    },
                ]
            },
        ).execute()


# ✅ ログ記録関数（ここに追加）
def log_chat_history(user_q, answer, source_type, is_unanswered):
    try:
        insert_log_row(
            CHAT_LOG_SHEET,
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_q.strip(),
                answer.strip(),
                source_type,
                str(is_unanswered).lower()
            ],
        )
    except Exception as e:
        print("❌ ログ出力失敗:", e)

GREETING_PATTERNS = ["こんにちは", "こんばんは", "おはよう", "はじめまして", "宜しくお願いします", "よろしくお願いします"]


def parse_json_request():
    if not request.is_json:
        return None, (jsonify({"error": "JSON形式で送信してください。"}), 415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "正しいJSON形式で送信してください。"}), 400)
    return data, None


def is_valid_session_id(session_id):
    return (
        isinstance(session_id, str)
        and bool(session_id.strip())
        and len(session_id) <= 128
        and all(char.isprintable() for char in session_id)
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "送信データが大きすぎます。内容を短くして再度お試しください。"}), 413


@app.errorhandler(429)
def rate_limit_exceeded(_error):
    return jsonify({"error": "リクエストが多すぎます。時間をおいて再度お試しください。"}), 429

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute; 100 per hour")
def chat():
    try:
        data, error_response = parse_json_request()
        if error_response:
            return error_response

        question = data.get("question", "")
        session_id = data.get("session_id")

        if not isinstance(question, str):
            return jsonify({"error": "質問は文字列で入力してください。"}), 400
        user_q = question.strip()

        if not user_q:
            return jsonify({"error": "質問がありません"}), 400
        if len(user_q) > 2000:
            return jsonify({"error": "質問は2000文字以内で入力してください。"}), 400
        if not is_valid_session_id(session_id):
            return jsonify({"error": "有効なセッションIDを指定してください。"}), 400

        if any(greet in user_q for greet in GREETING_PATTERNS):
            reply = "こんにちは！ご質問があればお気軽にどうぞ。"
            add_to_session_history(session_id, "assistant", reply)
            return jsonify({
                "response": reply,
                "original_question": user_q,
                "expanded_question": user_q
            })

        session_history = list(get_session_history(session_id))
        add_to_session_history(session_id, "user", user_q)

        # === クエリの種類に応じてリライト関数を自動選択 + ベクトル検索対象を決定 ===
        if is_reserve_query(user_q):
            expanded_q = expand_reserve_query(user_q, session_history)
            use_reserve = True
        else:
            expanded_q = expand_query(user_q, session_history)
            use_reserve = False

        q_vector = get_embedding(expanded_q)

        if use_reserve:
            D, I = reserve_index.search(np.array([q_vector]), k=7)
            search_source_flags = reserve_source_flags
            search_faq_questions = reserve_faq_questions
            search_faq_answers = reserve_faq_answers
            search_knowledge_contents = reserve_knowledge_contents
        else:
            D, I = index.search(np.array([q_vector]), k=7)
            search_source_flags = source_flags
            search_faq_questions = faq_questions
            search_faq_answers = faq_answers
            search_knowledge_contents = knowledge_contents


        faq_context = []
        reference_context = []

        for idx in I[0]:
            if idx < 0 or idx >= len(search_source_flags):
                continue
            src = search_source_flags[idx]
            if src == "faq":
                q = search_faq_questions[idx]
                a = search_faq_answers[idx]
                faq_context.append(f"Q: {q}\nA: {a}")
            elif src == "knowledge":
                ref_idx = idx - len(search_faq_questions)
                if ref_idx < len(search_knowledge_contents):
                    reference_context.append(f"【参考知識】{search_knowledge_contents[ref_idx]}")

        film_info_text = ""
        if not use_reserve:
            film_match_data = pf_matcher.match(user_q, session_history)
            film_info_text = pf_matcher.format_match_info(film_match_data)
        if film_info_text:
            reference_context.insert(0, film_info_text)

        if metadata_note:
            reference_context.append(f"【参考ファイル情報】{metadata_note}")

        if not faq_context and not reference_context and not film_info_text.strip():
            answer = (
                "当社はコーヒー製品の委託加工を専門とする会社です。"
                "恐れ入りますが、ご質問内容が当社業務と直接関連のある内容かどうかをご確認のうえ、"
                "改めてお尋ねいただけますと幸いです。\n\n"
                "ご不明な点がございましたら、当社の【お問い合わせフォーム】よりご連絡ください。"
            )
            add_to_session_history(session_id, "assistant", answer)
            return jsonify({
                "response": answer,
                "original_question": user_q,
                "expanded_question": expanded_q
            })

        faq_part = "\n\n".join(faq_context[:3]) if faq_context else "該当するFAQは見つかりませんでした。"
        ref_texts = [text for text in reference_context if "製品フィルム・カラー情報" in text]
        other_refs = [text for text in reference_context if "製品フィルム・カラー情報" not in text][:2]
        ref_part = "\n".join(ref_texts + other_refs)

        mode = infer_response_mode(user_q)

        prompt = f"""以下は当社のFAQおよび参考情報です。これらを参考に、ユーザーの質問に製造元の立場でご回答ください。

【FAQ】
{faq_part}

【参考情報】
{ref_part}

ユーザーの質問: {user_q}
回答："""

        system_prompt = base_prompt
        if mode == "short":
            system_prompt += "\n\n可能な限り簡潔かつ要点のみで回答してください。"
        elif mode == "long":
            system_prompt += "\n\n詳細な説明や具体例を含めて丁寧に回答してください。"

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()

        if "申し訳" in answer or "恐れ入りますが" in answer or "エラー" in answer:
            try:
                insert_log_row(
                    UNANSWERED_SHEET,
                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_q, "未回答", 1],
                )
            except Exception:
                print("[ERROR writing unanswered log]")
                traceback.print_exc()

        add_to_session_history(session_id, "assistant", answer)

        # ✅ 回答ソース・未回答判定・ログ出力をここに追加
        if use_reserve:
            source_type = "reserve_faq" if "Q:" in faq_part else "reserve_knowledge"
        else:
            source_type = "faq" if "Q:" in faq_part else "knowledge"

        is_unanswered = any(phrase in answer for phrase in ["申し訳", "恐れ入りますが", "エラー"])
        log_chat_history(user_q, answer, source_type, is_unanswered)

        return jsonify({
            "response": answer,
            "original_question": user_q,
            "expanded_question": expanded_q
        })

    except RequestEntityTooLarge:
        raise
    except Exception:
        print("[ERROR in /chat]")
        traceback.print_exc()
        return jsonify({
            "response": "一時的なエラーが発生しました。時間をおいて再度お試しください。"
        }), 500

@app.route("/feedback", methods=["POST"])
@limiter.limit("20 per minute; 200 per hour")
def feedback():
    data, error_response = parse_json_request()
    if error_response:
        return error_response

    question = data.get("question")
    answer = data.get("answer")
    feedback_value = data.get("feedback")
    reason = data.get("reason", "")

    if not all(isinstance(value, str) for value in [question, answer, feedback_value, reason]):
        return jsonify({"error": "フィードバック項目は文字列で送信してください。"}), 400

    question = question.strip()
    answer = answer.strip()
    feedback_value = feedback_value.strip()
    reason = reason.strip()

    if not all([question, answer, feedback_value]):
        return jsonify({"error": "不完全なフィードバックデータです"}), 400
    if (len(question) > 2000 or len(answer) > 10000
            or len(feedback_value) > 100 or len(reason) > 2000):
        return jsonify({"error": "フィードバックの入力文字数が上限を超えています。"}), 400

    try:
        insert_log_row(
            FEEDBACK_SHEET,
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, answer, feedback_value, reason],
        )
    except Exception:
        print("[ERROR writing feedback]")
        traceback.print_exc()
        return jsonify({"error": "フィードバックを保存できませんでした。時間をおいて再度お試しください。"}), 503

    return jsonify({"status": "success"})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running."

if __name__ == "__main__":
    app.run(debug=True)
