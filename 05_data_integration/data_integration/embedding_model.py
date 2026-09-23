from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # This is a very fast, highly accurate local embedding model
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts):
        # Convert an array of text strings into high-dimensional numerical vectors
        return self.model.encode(texts, show_progress_bar=True)
