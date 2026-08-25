import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# --- IMPORTY LANGCHAIN ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Klienci natywni (do embeddingów i bazy wektorowej)
openai_client = OpenAI()
qdrant_client = QdrantClient(path="data/qdrant_db")
COLLECTION_NAME = "owu_collection"

# Inicjalizacja modelu LLM przez LangChain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def ask_rag(query: str) -> dict:
    # 1. Wektoryzacja pytania (Zostawiamy natywne API OpenAI)
    query_vector = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    
    # 2. Wyszukiwanie w Qdrant
    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="text-dense",
        limit=10
    )
    
    context_texts = [hit.payload["page_content"] for hit in search_response.points]
    context_combined = "\n\n---\n\n".join(context_texts)
    
    # ========================================================
    # SEKCJA DEBUGOWANIA - PODGLĄDAMY CO ZWRACA BAZA QDRANT
    # ========================================================
    print("\n[DEBUG] --- ZNALEZIONY KONTEKST Z QDRANTA ---")
    if context_combined.strip():
        print(context_combined)
    else:
        print("UWAGA: Qdrant zwrócił PUSTY kontekst! Baza może być pusta.")
    print("========================================================\n")
    
    # 3. Szablon promptu LangChain (Foremka)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Jesteś profesjonalnym i precyzyjnym asystentem ds. ubezpieczeń.
        Twoim zadaniem jest odpowiedzieć na pytanie użytkownika w oparciu WYŁĄCZNIE o dostarczony poniżej kontekst z OWU (Ogólnych Warunków Ubezpieczenia).
        
        ZASADY:
        1. Jeśli odpowiedź nie znajduje się w kontekście, powiedz wprost: "Nie znalazłem tej informacji w dostarczonych fragmentach dokumentu." Nie zmyślaj.
        2. Odpowiadaj rzeczowo, jeśli to pomaga używaj punktów.
        
        KONTEKST:
        {context}"""),
        ("user", "{user_query}")
    ])
    
    # KROK A: Wypełniamy foremki danymi (wstrzykiwanie kontekstu i pytania)
    messages = prompt_template.invoke({
        "context": context_combined, 
        "user_query": query
    })
    
    # KROK B: Wysyłamy gotowe wiadomości do modelu
    ai_response = llm.invoke(messages)
    
    # KROK C: Wyciągamy sam czysty tekst z odpowiedzi
    answer = ai_response.content
    
    return {
        "answer": answer,
        "contexts": context_texts
    }

def main():
    print("\n" + "="*50)
    print("[INFO] RAG SYSTEM FOR OWU INITIALIZED (Powered by LangChain)")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")
    
    while True:
        query = input("\n[INPUT] Your query: ").strip()
        
        if query.lower() == 'exit':
            print("[INFO] Exiting program...")
            break
        if not query:
            continue
            
        print("[INFO] Generating response...")
        result = ask_rag(query)
        
        print(f"\n[RESPONSE]:\n{result['answer']}")
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()