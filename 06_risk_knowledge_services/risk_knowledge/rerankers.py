from sentence_transformers import CrossEncoder
from config import config

class Reranker:
    def __init__(self):
        # CrossEncoder acts as the strict "editor" to grade search results
        self.model = CrossEncoder(config.RERANKER_MODEL_NAME, max_length=512)
        
    def rerank(self, query, chunks, top_k=config.FINAL_TOP_K):
        if not chunks:
            return []
            
        # Create pairs of (user query, potential answer chunk)
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # The AI deeply reads the query and the text together to score relevance
        scores = self.model.predict(pairs)
        
        # Sort chunks by highest score
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the absolute best top_k chunks
        return [chunk for chunk, score in scored_chunks[:top_k]]
