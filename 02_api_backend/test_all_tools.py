import test_bootstrap  # noqa: F401
from decision_engine.generator import Generator

g = Generator()
queries = [
    "What is the weather in Sapporo?",
    "Are there any disaster warnings in Hokkaido?",
    "Are there any train delays on JR Hokkaido?"
]

print("=== VERIFYING ALL TOOLS ===")
for q in queries:
    print(f"\nQuery: '{q}'")
    res = g.generate(q, [], enabled_agents={"weather": True, "disaster": True, "train": True})
    print(f"Response: {res[:150]}...") # Truncated for readability
