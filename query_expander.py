import openai
import os

# 明示的に APIキー を設定（app.pyで設定済みなら不要）
openai.api_key = os.getenv("OPENAI_API_KEY")

def expand_query(user_input, session_history):
    try:
        if not session_history:
            return user_input

        context = session_history[-4:]
        context_text = "\n".join([f"{m['role']}: {m['content']}" for m in context])

        prompt = [
            {
                "role": "system",
                "content": "あなたは、ユーザーのあいまいな質問を、FAQ検索に最適な形式に言い換えるアシスタントです。意味を変えず、キーワードを補って明確な文章にしてください。"
            },
            {
                "role": "user",
                "content": f"""以下は直前のやり取りです：

{context_text}

この流れをふまえ、ユーザーの以下の質問を、FAQ検索に適したわかりやすい文に書き換えてください。

ユーザーの質問：「{user_input}」

→ 言い換え後：
"""
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.3,
            max_tokens=100
        )

        return (response.choices[0].message.content or "").strip() or user_input

    except Exception as e:
        print("❌ query_expander error:", e)
        return user_input
