import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API kaha hai laude...")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"
