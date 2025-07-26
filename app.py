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

EMBED_MODEL = "text-embedding-3-small"
VECTOR_PATH = "data/vector_data.npy"
INDEX_PATH = "data/index.faiss"

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
UNANSWERED_SHEET = "faq_suggestions"
FEEDBACK_SHEET = "feedback_log"
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

        # === クエリの種類に応じてリライト関数を自動選択 ===
        lower_q = user_q.lower()
        if any(x in lower_q for x in ["予約", "ログイン", "マニュアル", "アカウント", "登録"]):
            expanded_q = expand_reserve_query(user_q, session_history)
        else:
            expanded_q = expand_query(user_q, session_history)

        q_vector = get_embedding(expanded_q)

        D, I = index.search(np.array([q_vector]), k=7)
        if I.shape[1] == 0:
            raise ValueError("検索結果が見つかりませんでした")

        faq_context = []
        reference_context = []

        for idx in I[0]:
            if idx >= len(source_flags):
                continue
            src = source_flags[idx]
            if src == "faq":
                q = faq_questions[idx]
                a = faq_answers[idx]
                faq_context.append(f"Q: {q}\nA: {a}")
            elif src == "knowledge":
                ref_idx = idx - len(faq_questions)
                if ref_idx < len(knowledge_contents):
                    reference_context.append(f"【参考知識】{knowledge_contents[ref_idx]}")

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
            sheet_service.values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=f"{UNANSWERED_SHEET}!A2:D",
                valueInputOption="RAW",
                body={"values": [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_q, "未回答", 1]]}
            ).execute()

        add_to_session_history(session_id, "assistant", answer)

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
