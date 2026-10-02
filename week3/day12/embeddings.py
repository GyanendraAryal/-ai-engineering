import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


model = SentenceTransformer("all-MiniLM-L6-v2")  # 784
# text = "Machine learning is fun"

# embedding = model.encode(text)
# print(embedding[:10])
# print("================")
# print(embedding.shape)


t1 = "There are 24 paid leaves"
t2 = "Human head weighs 8 pounds"


v1 = model.encode(t1)
v2 = model.encode(t2)
print(cosine_similarity(v1, v2))
