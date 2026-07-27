import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def main():
    input_path = "data/processed/cleaned_owu.md"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}. Ensure 2_cleaning.py has been executed first.")
        
    print("[INFO] Loading cleaned Markdown file...")
    with open(input_path, "r", encoding="utf-8") as f:
        clean_markdown = f.read()

    print("[INFO] Stage 1: Splitting document by headers...")
    headers_to_split_on = [
        ("#", "Tytuł Główny"),
        ("##", "Paragraf"), 
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    logical_splits = markdown_splitter.split_text(clean_markdown)
    
    print("[INFO] Stage 2: Fallback chunking for long fragments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )
    final_splits = text_splitter.split_documents(logical_splits)

    print("[INFO] Stage 3: Extracting page numbers, cleaning tags, and enriching metadata...")
    
    current_page = "1"
    
    for chunk in final_splits:

        pages_found = re.findall(r'<!--\s*PAGE:\s*(\d+)\s*-->', chunk.page_content)
        
        if pages_found:
            chunk.metadata["Strona"] = ", ".join(list(dict.fromkeys(pages_found)))
            current_page = pages_found[-1]
        else:
            chunk.metadata["Strona"] = current_page
            
        chunk.page_content = re.sub(r'<!--\s*PAGE:\s*\d+\s*-->\n*', '', chunk.page_content)
        
        context_prefix = " | ".join([f"{k}: {v}" for k, v in chunk.metadata.items() if v])
        if context_prefix:
            chunk.page_content = f"[{context_prefix}]\n{chunk.page_content}"

    print(f"\n[INFO] CHUNKING STATISTICS:")
    print(f"- Logical splits count: {len(logical_splits)}")
    print(f"- Final enriched chunks count: {len(final_splits)}")
    
    max_len = max(len(chunk.page_content) for chunk in final_splits)
    print(f"- Maximum chunk length: {max_len} characters")
    
    print("\n--- FIRST ENRICHED CHUNK PREVIEW ---")
    print(f"Metadata: {final_splits[1].metadata}")
    print(f"Content:\n{final_splits[1].page_content[:300]}...")

if __name__ == "__main__":
    main()