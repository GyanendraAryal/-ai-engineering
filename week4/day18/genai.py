import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhaii")

client = genai.Client(api_key=api_key)
model = "gemini-flash-latest"
response = client.models.generate_content(
    model=model,
    contents="Explain how AI works in a few words",
)

print(response.text)
