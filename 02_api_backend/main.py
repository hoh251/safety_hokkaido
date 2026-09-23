from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import configure_module_paths

configure_module_paths()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our newly built RAG Pipeline
from agent_core.pipeline import RAGPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your Next.js frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We initialize the pipeline globally so it stays in RAM.
# This prevents the AI models from reloading on every single chat message!
try:
    pipeline = RAGPipeline()
except Exception as e:
    print(f"Warning: Could not initialize RAG Pipeline. Did you run build_index.py? Error: {e}")
    pipeline = None

from typing import Optional, List, Dict
class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    enabled_agents: Optional[Dict[str, bool]] = {"weather": True, "disaster": True, "train": True}

@app.get("/")
def read_root():
    return {"status": "RAG Backend is running"}

@app.post("/ask")
def ask_ai(req: ChatRequest):
    if pipeline is None:
         return {"reply": "Backend Error: The RAG Pipeline is not initialized. Please run `python build_index.py` first."}
         
    try:
        # DL05 Contextual Handling: Next.js sends the full 'messages' array
        if req.messages and len(req.messages) > 0:
            latest_message = req.messages[-1].get("content", "")
            answer = pipeline.ask(latest_message, chat_history=req.messages, enabled_agents=req.enabled_agents)
            return {"reply": answer}
            
        # Legacy fallback
        elif req.message:
            answer = pipeline.ask(req.message, enabled_agents=req.enabled_agents)
            return {"reply": answer}
            
        else:
            raise HTTPException(status_code=400, detail="Empty query provided.")
    except HTTPException:
        raise
    except Exception as e:
        return {"reply": f"Backend Error: {e}"}

@app.post("/reset_memory")
def reset_memory():
    from agent_core.memory import global_memory
    global_memory.clear()
    return {"status": "Memory wiped."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
