import faiss
import pickle
import json
import os
import numpy as np
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self, db_dir="vector_db"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Where we will save the indexes
        self.faiss_path = os.path.join(self.db_dir, "document.index")
        self.bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        self.chunk_store_path = os.path.join(self.db_dir, "chunk_store.json")

    def build_indexes(self, chunks, embeddings):
        print("=> Building FAISS dense index (Semantic Meaning)...")
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(faiss_index, self.faiss_path)

        print("=> Building BM25 sparse index (Exact Keyword Match)...")
        # Tokenize text for BM25
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(bm25, f)

        print("=> Saving chunk metadata mapping...")
        with open(self.chunk_store_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        print(f"Success! {len(chunks)} chunks saved to {self.db_dir}/")
