import json
import os
import glob
from data_integration.text_splitter import TextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

def load_all_data(data_dir: str):
    chunks = []
    splitter = TextSplitter()
    chunk_id_counter = 0

    # Grab all JSON, TXT, and PDF files
    all_files = []
    for ext in ["*.json", "*.txt", "*.pdf"]:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))

    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        if filename == "golden_set.json":
            continue

        # --- 1. Handle Structured JSON ---
        if filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping {filename} - Invalid JSON")
                    continue
                
                for category, situations in data.items():
                    for sit in situations:
                        situation_name = sit.get("situation", "General")
                        url = sit.get("url", "Unknown Source")
                        advices = sit.get("advices", [])
                        
                        context_header = f"Topic: {category}. Situation: {situation_name}. Advice: "
                        combined_text = context_header + " ".join(advices)
                        
                        split_texts = splitter.split_text(combined_text)
                        for i, split_text in enumerate(split_texts):
                            chunks.append({
                                "chunk_id": f"{filename}_{chunk_id_counter}_part_{i}",
                                "text": split_text,
                                "metadata": {
                                    "source_file": filename,
                                    "category": category,
                                    "situation": situation_name,
                                    "url": url
                                }
                            })
                        chunk_id_counter += 1

        # --- 2. Handle Unstructured TXT ---
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
            split_texts = splitter.split_text(raw_text)
            for i, split_text in enumerate(split_texts):
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id_counter}_part_{i}",
                    "text": split_text,
                    "metadata": {
                        "source_file": filename,
                        "category": "Unstructured Text",
                        "situation": "General Context",
                        "url": "Local Document"
                    }
                })
            chunk_id_counter += 1
            
        # --- 3. Handle Unstructured PDF ---
        elif filename.endswith(".pdf"):
            if PdfReader is None:
                print(f"Skipping {filename} - 'pypdf' is not installed.")
                continue
                
            try:
                reader = PdfReader(file_path)
                raw_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
                        
                split_texts = splitter.split_text(raw_text)
                for i, split_text in enumerate(split_texts):
                    chunks.append({
                        "chunk_id": f"{filename}_{chunk_id_counter}_part_{i}",
                        "text": split_text,
                        "metadata": {
                            "source_file": filename,
                            "category": "Unstructured PDF",
                            "situation": "General Context",
                            "url": "Local Document"
                        }
                    })
                chunk_id_counter += 1
            except Exception as e:
                print(f"Failed to read PDF {filename}: {e}")

    return chunks
