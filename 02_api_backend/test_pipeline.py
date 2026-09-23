import test_bootstrap  # noqa: F401
from agent_core.pipeline import RAGPipeline
pipeline = RAGPipeline()
print(pipeline.ask("What is the weather in Sapporo?", use_agent=False))
