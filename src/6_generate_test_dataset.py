import os
import json
import random
import shutil
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

def main():
    load_dotenv()

    NUMBER_OF_SEEDS = 10 
    random.seed(1)
    
    jsonl_path = "data/processed/chunks.jsonl" 
    
    all_chunks = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    chunk = data.get("page_content", "")
                    if chunk.strip():
                        all_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"[ERROR] File not found: {jsonl_path}.")
        return

    if len(all_chunks) < 3:
        print("[WARNING] Too few chunks in the file.")
        return

    valid_indices = list(range(1, len(all_chunks) - 1))
    selected_seeds = random.sample(valid_indices, NUMBER_OF_SEEDS)
    
    indices_to_extract = set()
    for idx in selected_seeds:
        indices_to_extract.add(idx - 1)
        indices_to_extract.add(idx)
        indices_to_extract.add(idx + 1)
        
    print(f"[INFO] Selected {NUMBER_OF_SEEDS} starting points. Total unique chunks: {len(indices_to_extract)}")

    temp_dir = "data/tmp_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for chunk_idx in indices_to_extract:
        file_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(all_chunks[chunk_idx])
        file_paths.append(file_path)

    print("[INFO] Initiating question generation via OpenAI...")
    synthesizer = Synthesizer()
    
    synthesizer.generate_goldens_from_docs(
        document_paths=file_paths,
        max_goldens_per_context=2 
    )

    dataset_path = "data/synthetic_dataset.json"
    
    goldens_data = []
    for g in synthesizer.synthetic_goldens:
        goldens_data.append({
            "input": g.input,
            "expected_output": g.expected_output,
            "context": g.context
        })
        
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(goldens_data, f, ensure_ascii=False, indent=4)
    
    shutil.rmtree(temp_dir)
    print(f"\n[SUCCESS] Dataset successfully saved to: {dataset_path}")

if __name__ == "__main__":
    main()