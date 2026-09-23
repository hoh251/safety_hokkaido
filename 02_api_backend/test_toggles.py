import test_bootstrap  # noqa: F401
from decision_engine.generator import Generator

g = Generator()
query = "What is the weather in Sapporo?"

res2 = g.generate(query, [], enabled_agents={"weather": True, "disaster": False, "train": False})
print("TEST 3 (Weather On, with city):", res2)
