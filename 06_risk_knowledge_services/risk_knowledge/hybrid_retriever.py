import numpy as np
import faiss
import pickle
import json
from config import config

class HybridRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        
        # Load the compiled databases from Phase 2
        self.faiss_index = faiss.read_index(config.FAISS_PATH)
        
        with open(config.BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        with open(config.CHUNK_STORE_PATH, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

    def retrieve(self, query, top_k=config.RETRIEVER_TOP_K):
        # 1. FAISS Dense Search (Finds meaning, e.g. "I'm cold" -> "Blizzard warning")
        query_vector = self.embedder.encode([query])
        D, I = self.faiss_index.search(np.array(query_vector).astype('float32'), top_k)
        
        dense_results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx != -1:
                dense_results.append((idx, float(D[0][i])))
                
        # If Hybrid toggle is off, just return FAISS results immediately
        if not config.USE_HYBRID:
            return [self.chunks[idx] for idx, score in dense_results]

        # 2. BM25 Sparse Search (Finds exact keywords, e.g. "Niseko")
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(bm25_scores)[::-1][:top_k]
        
        sparse_results = []
        for idx in top_n:
            sparse_results.append((idx, bm25_scores[idx]))
            
        # 3. Reciprocal Rank Fusion (RRF) - Combines Dense and Sparse results fairly
        rrf_k = 60
        fused_scores = {}
        
        for rank, (idx, _) in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
            
        for rank, (idx, _) in enumerate(sparse_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
            
        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        final_indices = sorted_indices[:top_k]
        
        return [self.chunks[idx] for idx in final_indices]
