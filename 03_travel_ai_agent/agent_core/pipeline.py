from data_integration.embedding_model import EmbeddingModel
from risk_knowledge.hybrid_retriever import HybridRetriever
from risk_knowledge.rerankers import Reranker
from decision_engine.generator import Generator
from agent_core.query_transform import QueryTransformer
from agent_core.memory import global_memory
from agent_core.router import Router
from config import config


class RAGPipeline:
    def __init__(self):
        print("Initializing Hokkaido RAG Pipeline components...")
        self.embedder = EmbeddingModel()
        self.retriever = HybridRetriever(self.embedder)

        # Respect the toggle switches in config.py
        if config.USE_RERANK:
            self.reranker = Reranker()
        else:
            self.reranker = None

        self.generator = Generator()
        self.transformer = QueryTransformer()
        self.router = Router()          # DL06: AI Router / Agent
        print("RAG Pipeline is online and ready!")

    def ask(self, query: str, chat_history=None, enabled_agents=None):

        # ── MEMORY: load conversation history ───────────────────────────────
        if config.USE_MEMORY:
            chat_history = global_memory.get_history()

        # ── DL06 ROUTER: classify intent before doing any heavy work ────────
        route_result = self.router.classify(query, chat_history)
        route = route_result["route"]   # "general" | "rag" | "realtime" | "rag+realtime"

        # ── DL05 QUERY REFORMULATION: make standalone if needed ─────────────
        if config.USE_MEMORY and chat_history:
            standalone_query = self.transformer.reformulate_query(query, chat_history)
            print(f"[Reformulation] '{query}' → '{standalone_query}'")
        else:
            standalone_query = query

        # ── LANGUAGE: translate to English for retrieval ─────────────────────
        english_query = self.transformer.translate_to_english(standalone_query)
        print(f"[Language] '{standalone_query}' → '{english_query}'")

        # ── RAG RETRIEVAL: only run if the router says we need docs ─────────
        final_chunks = []
        if route in ("rag", "rag+realtime"):
            print(f"[Pipeline] Route='{route}' → Activating RAG retrieval.")
            candidates = self.retriever.retrieve(english_query, top_k=config.RETRIEVER_TOP_K)
            if self.reranker and config.USE_RERANK:
                final_chunks = self.reranker.rerank(english_query, candidates, top_k=config.FINAL_TOP_K)
            else:
                final_chunks = candidates[:config.FINAL_TOP_K]
        else:
            print(f"[Pipeline] Route='{route}' → Skipping RAG retrieval.")

        # ── TOOLS: only activate real-time tools if the router says so ──────
        if route in ("realtime", "rag+realtime"):
            print(f"[Pipeline] Route='{route}' → Activating real-time tools.")
            active_agents = {"weather": True, "disaster": True, "train": True}
        else:
            print(f"[Pipeline] Route='{route}' → Skipping real-time tools.")
            active_agents = {"weather": False, "disaster": False, "train": False}

        # Allow the API caller to still override tools (e.g. frontend toggles)
        if enabled_agents is not None:
            for key in enabled_agents:
                if not enabled_agents[key]:
                    active_agents[key] = False

        # ── GENERATE: send chunks + history to Groq ─────────────────────────
        if config.USE_MEMORY:
            full_history = global_memory.get_history() + [{"role": "user", "content": query}]
        else:
            full_history = chat_history if chat_history else [{"role": "user", "content": query}]

        answer = self.generator.generate(query, final_chunks, full_history, active_agents)

        # ── MEMORY: save exchange ────────────────────────────────────────────
        if config.USE_MEMORY:
            global_memory.add_user_message(query)
            global_memory.add_ai_message(answer)

        return answer
