from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

client = OpenAI(api_key=api_key)
model = "text-embedding-3-small"
text = "Eiffel Tower is in Paris and is a famous landmark, it is 325 meters tall"
response = client.embeddings.create(
    input=text,
    model=model,
)
print("Vector Embeddings: ", response.data[0].embedding)
