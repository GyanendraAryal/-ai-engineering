import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API kaha khai bhai")

client = Groq(api_key=api_key)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_model = "openai/gpt-oss-120b"


documents = [
    "Employees receive 24 days of paid leave per year",
    "Employees work from the office on Tuesday, Wednesday and Thursday.",
    "Employees receive 4000 per month for gym reimbursement",
    "Employees can claim 2000 per month for home internet",
    "Employees can have 90 days notice period.",
]


# Create embeddings for all documents
document_embeddings = embedding_model.encode(documents)
# print(document_embeddings.shape)
print(f"Size of doc:  {sys.getsizeof(document_embeddings)} bytes")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query_embedding):
    scores = []

    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))

    scores.sort(reverse=True)

    return scores[0]


def ask_llm(question, context):
    system_prompt = system_prompt = f"""
Answer the user's question using only the provided context.

Context:
{context}

If the answer cannot be found in the context, say:
"I don't have enough information."

Answer in one line.
Do not hallucinate.
"""
    system_message = {
        "role": "system",
        "content": system_prompt,
    }
    message = {
        "role": "user",
        "content": question,
    }
    messages = [message, system_message]
    response = client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer


query = "How much vacation do I get?"
query_embedding = embedding_model.encode(query)
score, context = retrieve(query_embedding)
answer = ask_llm(query, context)
print(answer)
