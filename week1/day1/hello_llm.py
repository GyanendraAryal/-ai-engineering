import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API key kaha hai bhai")
client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"
role = "user"
content = "Do you know Pratyush"
message = {"role": role, "content": content}
messages = [message]
response = client.chat.completions.create(model=model, messages=messages)
# print(response)
answer = response.choices[0].message.content
print(answer)
