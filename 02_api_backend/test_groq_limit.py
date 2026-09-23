import test_bootstrap  # noqa: F401
from agent_core.pipeline import RAGPipeline
rag = RAGPipeline()
print(rag.ask("ตอนนี้สภาพอากาศที่ซัปโปโรเป็นอย่างไรบ้าง?"))
