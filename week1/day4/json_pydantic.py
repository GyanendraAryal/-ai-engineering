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

# Structure
from pydantic import BaseModel


class Ticket(BaseModel):
    name: str
    email: str
    phone: str
    issue: str


schema = Ticket.model_json_schema()
response_format = {
    "type": "json_object",
}
system_prompt = f"""
Extract the personal information from the ticket and strictly based on this schema and give a json output {schema}
"""
message_system = {
    "role": "system",
    "content": system_prompt,
}

content = "Hello My name is Gyanendra. I have iphone which is not working at all. My address is delhi and my father's name is Tulshi. My contact number is 9805497816"
prompt = f"""
This is a customer ticket. Please extract personal information from this. {content}
"""
message = {
    "role": role,
    "content": prompt,
}
messages = [message_system, message]
response = client.chat.completions.create(
    model=model, messages=messages, response_format=response_format
)
answer = response.choices[0].message.content
# print(answer)
# Now reading JSON
import json

raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)
# print(ticket.name)
# print(ticket.email)
# print(ticket.phone)
# print(ticket.issue)
print(ticket)
