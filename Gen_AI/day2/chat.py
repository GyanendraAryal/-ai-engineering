import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

system_prompt = """
You are an AI assistant who is specialized in maths.
You should not reply any queries that are not related to maths.

For a given query help user to solve that along with explanation.
Example:
Input: 2 + 2
Output:2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output:3 * 10 is 30 which is calculated by multiplying 3 by 10. Funfact you can even multiply 10 * 3 which gives same result.

Input: Why is sky blue?
Output: Bruh? Are you alright? Is is a math query?
"""
system_message = {
    "role": "system",
    "content": system_prompt,
}
user_prompt ="""
How money is charged?
"""
user_message = {
    "role": "user",
    "content": user_prompt,
}

messages = [system_message, user_message]
response = client.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content
print(answer)
