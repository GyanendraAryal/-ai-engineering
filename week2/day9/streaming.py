import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API kaha hai laude")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

prompt = "Explain how internet works"
message = {
    "role": "user",
    "content": prompt,
}
messages = [message]
stream = client.chat.completions.create(model=model, messages=messages, stream=True)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)


# response = client.chat.completions.create(
#     model=model,
#     messages=messages,
#     temperature=0,
# )
# answer = response.choices[0].message.content
# print(answer)
