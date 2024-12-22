import os
import openai

# OpenAIのAPIキーを環境変数から取得

def ask_chatgpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # または "gpt-3.5-turbo"
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response["choices"][0]["message"]["content"]

# 質問内容を指定
question = "Pythonでのリスト内包表記の使い方を教えてください。"

# ChatGPTに質問して回答を取得
answer = ask_chatgpt(question)

# 回答を表示
print(answer)
