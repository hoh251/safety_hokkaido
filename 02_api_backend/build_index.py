import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import configure_module_paths

configure_module_paths()

from data_integration.document_loader import load_all_data
from data_integration.embedding_model import EmbeddingModel
from data_integration.vector_store import VectorStore
from data_integration.index_meta import IndexMeta
from config import config

def main():
    print("--- Checking Dataset Fingerprint ---")
    meta = IndexMeta(config.DATA_DIR, os.path.join(config.DB_DIR, "index_meta.json"))
    
    if not meta.is_stale():
        print("✅ Data has not changed since the last build. Skipping database generation to save time!")
        return
    else:
        print("⚠️ Data has changed (or this is the first run). Initiating full database build...")

    print("\n--- Phase 1: Loading & Chunking ---")
    chunks = load_all_data(config.DATA_DIR)
    if not chunks:
        print("Error: No data found in the 'data' directory!")
        return
    print(f"Loaded {len(chunks)} contextual chunks from files.")
    
    print("\n--- Phase 2: Generating Embeddings ---")
    print("Downloading/Loading SentenceTransformer Model...")
    embedder = EmbeddingModel()
    
    print("Converting text to vectors...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    
    print("\n--- Phase 3: Building Vector Store ---")
    store = VectorStore()
    store.build_indexes(chunks, embeddings)
    
    # Save the new fingerprint so we don't rebuild next time
    meta.update_meta()
    
    print("\n✅ DONE! The RAG backend is now primed with knowledge.")

if __name__ == "__main__":
    main()
