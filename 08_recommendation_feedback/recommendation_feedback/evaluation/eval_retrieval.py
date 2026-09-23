import json
import os
import sys

# Add parent directory to path so we can import src from inside the evaluation folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_integration.embedding_model import EmbeddingModel
from risk_knowledge.hybrid_retriever import HybridRetriever
from config import config
from recommendation_feedback.evaluation.metrics import calculate_hit_rate, calculate_mrr

from agent_core.query_transform import QueryTransformer

def run_retrieval_eval():
    print("Loading Golden Dataset...")
    golden_file = os.path.join(config.DATA_DIR, "golden_set.json")
    if not os.path.exists(golden_file):
        print(f"Error: Golden dataset not found at {golden_file}")
        return

    with open(golden_file, "r") as f:
        golden_data = json.load(f)

    print("Initializing FAISS Retriever & Translator...")
    embedder = EmbeddingModel()
    retriever = HybridRetriever(embedder)
    transformer = QueryTransformer()

    total_hit = 0
    total_mrr = 0

    print("\n--- Evaluating Retrieval (Database Accuracy) ---")
    for item in golden_data:
        question = item["question"]
        lang = item.get("language", "Unknown")
        expected_ids = item["expected_chunk_ids"]
        
        print(f"\n[{lang}] Q: {question}")
        
        # 1. Translate the query to English first!
        english_question = transformer.translate_to_english(question)
        
        # 2. Test TOP_K retrieval using the English query
        results = retriever.retrieve(english_question, top_k=config.RETRIEVER_TOP_K)
        
        hit = calculate_hit_rate(results, expected_ids)
        mrr = calculate_mrr(results, expected_ids)
        
        total_hit += hit
        total_mrr += mrr
        
        print(f"Hit@10: {hit} | MRR: {mrr:.2f}")

    avg_hit = total_hit / len(golden_data)
    avg_mrr = total_mrr / len(golden_data)

    print(f"\n--- Final Database Scores ---")
    print(f"Hit Rate (Did it find the answer?): {avg_hit * 100:.1f}%")
    print(f"MRR (Was it ranked at the top?):  {avg_mrr:.3f}")

if __name__ == "__main__":
    run_retrieval_eval()
