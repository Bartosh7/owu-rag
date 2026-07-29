import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

def main():
    load_dotenv()
    
    openai_client = OpenAI()
    qdrant_client = QdrantClient(path="data/qdrant_db")
    collection_name = "owu_collection"
    
    print("\n" + "="*50)
    print("[INFO] RAG SYSTEM FOR OWU INITIALIZED")
    print("Type your insurance-related queries.")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")
    
    while True:
        query = input("\n[INPUT] Your query: ").strip()
        
        if query.lower() == 'exit':
            print("[INFO] Exiting program...")
            break
            
        if not query:
            continue
            
        print("[INFO] Searching database...")
        query_vector = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="text-dense",
            limit=7
        )
        
        print("\n[INFO] Retrieved context fragments from database:")
        context_texts = []
        
        for i, hit in enumerate(search_response.points):
            score = round(hit.score, 3)
            content = hit.payload["page_content"]
            metadata = hit.payload.get("metadata", "No metadata")
            
            print(f"   [{i+1}] Score: {score} | Metadata: {metadata}")
            print(f"      \"{content[:150]}...\"")
            
            context_texts.append(content)
            
        context_combined = "\n\n---\n\n".join(context_texts)
        
        print("\n[INFO] Generating response...")
        system_prompt = f"""
        Jesteś profesjonalnym i precyzyjnym asystentem ds. ubezpieczeń.
        Twoim zadaniem jest odpowiedzieć na pytanie użytkownika w oparciu WYŁĄCZNIE o dostarczony poniżej kontekst z OWU (Ogólnych Warunków Ubezpieczenia).
        
        ZASADY:
        1. Jeśli odpowiedź nie znajduje się w kontekście, powiedz wprost: "Nie znalazłem tej informacji w dostarczonych fragmentach dokumentu." Nie zmyślaj.
        2. Odpowiadaj rzeczowo, jeśli to pomaga używaj punktów.
        
        KONTEKST:
        {context_combined}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        
        print(f"\n[RESPONSE]:\n{response.choices[0].message.content}")
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()