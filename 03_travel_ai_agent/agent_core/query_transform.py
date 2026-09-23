import requests
from config import config

class QueryTransformer:
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = config.LLM_MODEL
        
    def translate_to_english(self, query):
        """Translates a foreign query into English so the database can understand it perfectly."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional translator. You MUST translate the user's input into English. DO NOT answer the user's question. ONLY output the English translation."
                },
                {
                    "role": "user", 
                    "content": f"Translate this into English: {query}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            english_query = response.json()["choices"][0]["message"]["content"].strip()
            # If the LLM somehow fails and returns empty string, fallback to original query
            if not english_query:
                return query
            return english_query
        except Exception as e:
            print(f"Translation Error: {e}")
            return query

    def reformulate_query(self, query, chat_history):
        """Rewrites a contextual follow-up query into a Standalone Query using the chat history."""
        if not chat_history:
            return query
            
        history_str = ""
        for msg in chat_history[-4:]:
            role = "AI" if msg["role"] in ["ai", "assistant"] else "User"
            history_str += f"{role}: {msg['content']}\n"
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Given the following conversation history and the user's follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the history. DO NOT answer the question. ONLY output the standalone question."
                },
                {
                    "role": "user", 
                    "content": f"Chat History:\n{history_str}\n\nFollow-up question: {query}\n\nStandalone question:"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            standalone = response.json()["choices"][0]["message"]["content"].strip()
            if not standalone:
                return query
            return standalone
        except Exception as e:
            print(f"Reformulation Error: {e}")
            return query
