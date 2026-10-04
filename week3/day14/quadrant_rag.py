import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Distance:- for cosine similarity
# VectorParams:- kind of instruction manual for vector DB
# PointStruct:- to create points
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QUDRANT_URL = os.getenv("QUDRANT_URL")
QUDRANT_API_KEY = os.getenv("QUDRANT_API_KEY")

# 1 Connect to Qdrant
client = QdrantClient(
    url=QUDRANT_URL,
    api_key=QUDRANT_API_KEY,
)
print("Connected to Qdrant cloud!")


# 2 Create Qdrant Collection
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

# Delete collection if already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting the existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)
print(f"Collection name: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print(f"Distance: {Distance.COSINE}")

# 3 Load our knowledge
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(documents)} documents")

# 4 Create Embeddings
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model is ready to go.")

embeddings = model.encode(documents)
print(f"Generate {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# 5 Create Qdrant size
points = []

for i, embeddings in enumerate(embeddings):
    point = PointStruct(
        id=i + 1,
        vector=embeddings.tolist(),
        payload={
            "text": documents[i],
        },
    )
    points.append(point)

# 6 Upload to Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)
print(f"Uploaded {len(points)} documents to Qdrant!")


# 7 Search Qdrant
def search(query, top_k=3):
    # convert the question into embeddings
    query_vector = model.encode(query).tolist()

    # search qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


# 8 Test Search
query = "How many vacations do i get?"
results = search(query, top_k=3)
print("\n Search results: ")
for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()

# 9 Connect to Qdrant
groq_client = Groq(api_key=GROQ_API_KEY)
groq_model = "openai/gpt-oss-120b"


# 10 Ask the llm
def ask_llm(question, context):
    prompt = f"""
    Answer the question using only the information provided below.
    Context: {context}
    Question: {question}
    If the answer is not present in the context, say:
    "I don't know based on the available information"
    """
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    answer = response.choices[0].message.content


# 11 Complete RAG pipeline
question = "How many vacation days do I get?"
results = search(question, top_k=3)
# Extract text from the search results
context = "\n".join(result.payload["text"] for result in results)
answer = ask_llm(question, context)
print("\n Final Answer: ")
print(answer)
