import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

# Step-1
knowledge_base = {
    "age": "The age of speed is 25 years",
    "net_worth": "The net worth of speed is a billion dollars",
}


# Step-2
def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net_worth" in question:
        return knowledge_base["net_worth"]
    else:
        return None


def ask_llm(question):
    context = retrieve_info(question)
    system_prompt = f"Answer in one line only. Answer only based on the context {context}. Do not hallucuinate"
    system_message = {
        "role": "system",
        "content": system_prompt,
    }
    message = {
        "role": "user",
        "content": question,
    }
    messages = [message, system_message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer


# question = "What is speed's age?"
question = "How rich is speed?"
print(ask_llm(question))
