# Intelligent Insurance Assistant (RAG System PoC)

A Proof of Concept Retrieval-Augmented Generation (RAG) system designed to accurately analyze complex insurance documents (OWU). The architecture is built with a strict "Zero Hallucination Tolerance" approach, grounding all LLM responses entirely in the retrieved context.

## Technologies Used
* Python 3
* OpenAI API (gpt-4o-mini, text-embedding-3-small)
* LangChain (imperative orchestration)
* Qdrant (local vector database)
* LlamaParse (Vision LLM for complex PDF parsing)
* DeepEval (Data-Centric automated evaluation)

## Data Pipeline
1. Data Ingestion & Sanitization: Parsing complex document tables with LlamaParse and removing visual noise using Regex.
2. Context-Aware Chunking: Two-stage splitting (Semantic Markdown + Recursive Character) with metadata injection.
3. Vectorization & Indexing: Batch processing of embeddings stored in Qdrant.
4. Search & Generation: LangChain-based prompt templates with strict anti-hallucination instructions.
5. Evaluation: Automated Golden Dataset generation via Sliding Window context and DeepEval metrics.

## Next Steps
Currently, the system relies on dense vector search. The next planned step is to implement hybrid search by integrating lexical retrieval (e.g., BM25) to improve accuracy for exact matches and domain-specific acronyms.

## Quickstart
To use the search functionality, you must first execute the data pipeline sequentially to process the documents and build the database.

1. Clone the repository and install dependencies.
2. Create a `.env` file and add your `LLAMA_CLOUD_API_KEY` and `OPENAI_API_KEY`.
3. Run the data pipeline scripts in order:
   ```bash
   uv run src/1_ingestion.py
   uv run src/2_cleaning.py
   uv run src/3_chunking.py
   uv run src/4_indexing.py