import openai
import os

# 環境変数からAPIキーを読み込み（app.pyで設定済みであればスキップ可）
openai.api_key = os.getenv("OPENAI_API_KEY")

def expand_reserve_query(user_input, session_history):
    try:
        if not user_input:
            return ""

        # 直近4件の履歴を取得
        context = session_history[-4:] if session_history else []
        context_text = "\n".join([f"{m['role']}: {m['content']}" for m in context])

        prompt = [
            {
                "role": "system",
                "content": (
                    "あなたは、予約システムに関するあいまいな質問を、FAQ検索に適した明確な文章に書き換えるアシスタントです。"
                    "意味を変えず、キーワードを補い、予約画面・機能名・操作手順が明確になるようにしてください。"
                    "言い換えた文章は、1文の日本語文で出力してください。"
                )
            },
            {
                "role": "user",
                "content": f"""以下は最近のやり取りです：

{context_text}

この流れをふまえ、ユーザーの以下の質問を予約システムFAQ検索用に明確な文に言い換えてください。

ユーザーの質問：「{user_input}」

→ 言い換え後：
"""
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.2,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ expand_reserve_query error:", e)
        return user_input
