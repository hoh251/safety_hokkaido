# Safety Hokkaido — Agentic AI System

Safety Hokkaido is an AI travel-safety assistant for Hokkaido. It combines a multilingual Next.js interface, a FastAPI service, local safety-document retrieval (RAG), real-time data tools, and a guarded LLM response layer.

## Architecture

```text
01_web_app → 02_api_backend → 03_travel_ai_agent
                                      ├→ 04_external_data_services
                                      ├→ 05_data_integration
                                      ├→ 06_risk_knowledge_services
                                      └→ 07_decision_llm_engine → 08_recommendation_feedback
```

| Module | Responsibility | Current implementation |
|---|---|---|
| `01_web_app` | Traveler-facing UI, localization, chat client | Next.js application |
| `02_api_backend` | HTTP API, validation, CORS, service startup | FastAPI `main.py` |
| `03_travel_ai_agent` | Intent routing and orchestration | `agent_core/` |
| `04_external_data_services` | Weather, disaster and rail adapters | `external_data/` |
| `05_data_integration` | Loading, splitting, embeddings and indices | `data_integration/` |
| `06_risk_knowledge_services` | Hybrid retrieval, reranking and verified safety corpus | `risk_knowledge/` |
| `07_decision_llm_engine` | Grounded generation and safety policy | `decision_engine/` |
| `08_recommendation_feedback` | Evaluation, evidence and future feedback loop | `recommendation_feedback/evaluation/` |

## Quick start

1. Configure the backend (PowerShell):

   ```powershell
   cd 02_api_backend
   Copy-Item .env.example .env
   # Add GROQ_API_KEY and METEOSOURCE_API_KEY to .env
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python build_index.py
   uvicorn main:app --reload --port 8000
   ```

2. In another terminal, start the web application:

   ```powershell
   cd 01_web_app
   Copy-Item .env.local.example .env.local
   npm install
   npm run dev
   ```

3. Open `http://localhost:3000`.

Read each module's `01_env.md`, `02_step.md`, and `03_process.md` before extending the implementation. These documents define module boundaries so safety decisions remain auditable.

