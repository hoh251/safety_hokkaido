import json
import os
import sys
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core.pipeline import RAGPipeline
from config import config

def evaluate_with_llm_judge(question, answer, context):
    """Uses Groq (LLM-as-a-judge) to score the generated answer."""
    
    judge_prompt = f"""You are an expert AI evaluator. Grade the Chatbot's Answer based ONLY on the provided Context.
    
    Question: {question}
    Context: {context}
    Chatbot Answer: {answer}
    
    Output ONLY a JSON object with two integer scores (1-10):
    {{
        "faithfulness_score": (1-10, did the chatbot use ONLY the context without hallucinating?),
        "relevance_score": (1-10, did the chatbot actually answer the user's question?)
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.LLM_MODEL, 
        "messages": [{"role": "user", "content": judge_prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0 # Strict, logical evaluation
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except:
        return {"faithfulness_score": 0, "relevance_score": 0}

def run_generation_eval():
    print("Loading Golden Dataset...")
    golden_file = os.path.join(config.DATA_DIR, "golden_set.json")
    with open(golden_file, "r") as f:
        golden_data = json.load(f)
        
    pipeline = RAGPipeline()
    
    total_faithfulness = 0
    total_relevance = 0
    
    print("\n--- Evaluating Generation (LLM-as-a-Judge) ---")
    for item in golden_data:
        question = item["question"]
        
        lang = item.get("language", "Unknown")
        print(f"\n[{lang}] Q: {question}")
        print("Generating answer...")
        
        # 1. Translate the query to English first!
        english_question = pipeline.transformer.translate_to_english(question)
        
        # Bypass pipeline.ask() so we can extract the exact context used
        candidates = pipeline.retriever.retrieve(english_question, top_k=config.RETRIEVER_TOP_K)
        if pipeline.reranker and config.USE_RERANK:
            final_chunks = pipeline.reranker.rerank(english_question, candidates, top_k=config.FINAL_TOP_K)
        else:
            final_chunks = candidates[:config.FINAL_TOP_K]
            
        # The generator receives the original foreign question so it answers in Thai
        answer = pipeline.generator.generate(question, final_chunks)
        context_str = "\n".join([c["text"] for c in final_chunks])
        
        print("Judging answer...")
        scores = evaluate_with_llm_judge(question, answer, context_str)
        
        f_score = scores.get("faithfulness_score", 0)
        r_score = scores.get("relevance_score", 0)
        
        print(f"Faithfulness: {f_score}/10 | Relevance: {r_score}/10")
        
        total_faithfulness += f_score
        total_relevance += r_score
        
    avg_f = total_faithfulness / len(golden_data)
    avg_r = total_relevance / len(golden_data)
    
    print(f"\n--- Final LLM Scores ---")
    print(f"Average Faithfulness (No Hallucination): {avg_f:.1f}/10")
    print(f"Average Relevance (Actually answered):   {avg_r:.1f}/10")

if __name__ == "__main__":
    run_generation_eval()
