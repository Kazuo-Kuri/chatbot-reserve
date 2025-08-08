# app.py
import os
import json
import time
import base64
from datetime import datetime
from dotenv import load_dotenv

# 🛡️ proxy 環境変数の削除
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

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

search_corpus = faq_questions + knowledge_contents
source_flags = ["faq"] * len(faq_questions) + ["knowledge"] * len(knowledge_contents)

# ✅ 予約システム用 FAQ の読み込み
with open("data/reserve_faq.json", encoding="utf-8") as f:
    reserve_faq_items = json.load(f)
reserve_faq_questions = [item["question"] for item in reserve_faq_items]
reserve_faq_answers = [item["answer"] for item in reserve_faq_items]

# ✅ 予約システム用 Knowledge の読み込み
with open("data/reserve_knowledge.json", encoding="utf-8") as f:
    reserve_knowledge_dict = json.load(f)
reserve_knowledge_contents = [
    f"{category}：{text}" for category, texts in reserve_knowledge_dict.items() for text in texts
]

# ✅ 予約システム用 検索対象とフラグ
reserve_search_corpus = reserve_faq_questions + reserve_knowledge_contents
reserve_source_flags = ["faq"] * len(reserve_faq_questions) + ["knowledge"] * len(reserve_knowledge_contents)

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

# --- 予約専用データの読み込み ---
with open("data/reserve_faq.json", "r", encoding="utf-8") as f:
    reserve_faq_list = json.load(f)

with open("data/reserve_knowledge.json", "r", encoding="utf-8") as f:
    reserve_knowledge_dict = json.load(f)

reserve_knowledge_texts = [
    f"{category}：{text}"
    for category, texts in reserve_knowledge_dict.items()
    for text in texts
]

reserve_corpus = [
    f"{item['question']} {item['answer']}" for item in reserve_faq_list
] + reserve_knowledge_texts

reserve_index = faiss.read_index("data/reserve_index.faiss")
# --- ここまで追加 ---

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

# 🔽 ここに予約用検索関数を追加

def search_reserve_knowledge(user_q, k=3):
    query_vector = get_embedding(user_q).astype("float32").reshape(1, -1)
    scores, indices = reserve_index.search(query_vector, k)
    hits = [reserve_corpus[i] for i in indices[0] if i < len(reserve_corpus)]
    return hits

# 🔽 ここに判定関数を追加
def is_reserve_query(user_q):
    keywords = ["予約", "納期", "製造日", "納品", "アクセス", "ID", "パスワード", "ログイン"]
    return any(kw in user_q for kw in keywords)

if os.path.exists(VECTOR_PATH) and os.path.exists(INDEX_PATH):
    vector_data = np.load(VECTOR_PATH)
    index = faiss.read_index(INDEX_PATH)
else:
    vector_data = np.array([get_embedding(text) for text in search_corpus], dtype="float32")
    index = faiss.IndexFlatL2(vector_data.shape[1])
    index.add(vector_data)
    np.save(VECTOR_PATH, vector_data)
    faiss.write_index(index, INDEX_PATH)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
UNANSWERED_SHEET = "faq_suggestions_reserve"
FEEDBACK_SHEET = "feedback_log_reserve"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

credentials_info = json.loads(base64.b64decode(os.environ["GOOGLE_CREDENTIALS"]).decode("utf-8"))
credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
sheet_service = build("sheets", "v4", credentials=credentials).spreadsheets()

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

# ✅ ログ記録関数（ここに追加）
def log_chat_history(user_q, answer, source_type, is_unanswered):
    try:
        sheet_service.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{CHAT_LOG_SHEET}!A2:E",
            valueInputOption="RAW",
            body={"values": [[
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_q.strip(),
                answer.strip(),
                source_type,
                str(is_unanswered).lower()
            ]]}
        ).execute()
    except Exception as e:
        print("❌ ログ出力失敗:", e)

GREETING_PATTERNS = ["こんにちは", "こんばんは", "おはよう", "はじめまして", "宜しくお願いします", "よろしくお願いします"]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_q = data.get("question", "").strip()
        session_id = data.get("session_id", "default")

        if not user_q:
            return jsonify({"error": "質問がありません"}), 400

        if any(greet in user_q for greet in GREETING_PATTERNS):
            reply = "こんにちは！ご質問があればお気軽にどうぞ。"
            add_to_session_history(session_id, "assistant", reply)
            return jsonify({
                "response": reply,
                "original_question": user_q,
                "expanded_question": user_q
            })

        add_to_session_history(session_id, "user", user_q)
        session_history = get_session_history(session_id)

        # === クエリの種類に応じてリライト関数を自動選択 + ベクトル検索対象を決定 ===
        lower_q = user_q.lower()
        if any(x in lower_q for x in ["予約", "ログイン", "マニュアル", "アカウント", "登録"]):
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

        D, I = index.search(np.array([q_vector]), k=7)
        if I.shape[1] == 0:
            raise ValueError("検索結果が見つかりませんでした")

        faq_context = []
        reference_context = []

        for idx in I[0]:
            if idx >= len(search_source_flags):
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
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()

        if "申し訳" in answer or "恐れ入りますが" in answer or "エラー" in answer:
            sheet_service.values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{UNANSWERED_SHEET}!A2:D",
                valueInputOption="RAW",
                body={"values": [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_q, "未回答", 1]]}
            ).execute()

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

    except Exception as e:
        print("[ERROR in /chat]:", e)
        return jsonify({
            "response": "エラーが発生しました。",
            "error": str(e)
        }), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    feedback_value = data.get("feedback")
    reason = data.get("reason", "")

    if not all([question, answer, feedback_value]):
        return jsonify({"error": "不完全なフィードバックデータです"}), 400

    sheet_service.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{FEEDBACK_SHEET}!A2:E",
        valueInputOption="RAW",
        body={"values": [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, answer, feedback_value, reason]]}
    ).execute()

    return jsonify({"status": "success"})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running."

if __name__ == "__main__":
    app.run(debug=True)