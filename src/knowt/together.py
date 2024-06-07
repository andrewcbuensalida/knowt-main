from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = os.environ.get("TOGETHER_AI_KEY")

client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant",
        },
        {
            "role": "user",
            "content": "Tell me about San Francisco",
        },
    ],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens=1024,
)

print(chat_completion.choices[0].message.content)
