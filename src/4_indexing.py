import os
import json
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

def main():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("[ERROR] Missing OPENAI_API_KEY in the .env file.")

    print("[INFO] Initializing OpenAI and Qdrant clients...")
    openai_client = OpenAI()
    
    qdrant_client = QdrantClient(path="data/qdrant_db")
    
    collection_name = "owu_collection"
    
    if not qdrant_client.collection_exists(collection_name):
        print(f"[INFO] Creating new collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            }
        )
    else:
        print(f"[INFO] Collection {collection_name} already exists.")

    input_path = "data/processed/chunks.jsonl"
    print(f"[INFO] Loading data from {input_path}...")
    
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"[INFO] Found {len(chunks)} fragments to vectorize.")

    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [item["page_content"] for item in batch]
        
        print(f"[INFO] Processing batch {i + 1} - {min(i + batch_size, len(chunks))} of {len(chunks)}...")
        
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        points = []
        for j, item in enumerate(batch):
            point_id = str(uuid.uuid4())
            vector = response.data[j].embedding
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"text-dense": vector},
                    payload={
                        "page_content": item["page_content"],
                        "metadata": item["metadata"]
                    }
                )
            )
            
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    print("\n[SUCCESS] All vectors have been successfully saved to the Qdrant database (data/qdrant_db).")

if __name__ == "__main__":
    main()