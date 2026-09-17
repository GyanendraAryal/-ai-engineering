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
content = "Suggest a name for my clothing company"
message_system = {
    "role": "system",
    "content": "You are a brand manager who suggest name for my brand. Only only word",
}
message = {"role": role, "content": content}
messages = [message_system, message]
# Temperatue ramge is [0,2]
response = client.chat.completions.create(model=model, messages=messages, temperature=2)
# print(response)
answer = response.choices[0].message.content
print(answer)
