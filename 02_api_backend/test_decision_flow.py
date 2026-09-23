"""Local, dependency-light contract tests for the safety decision flow."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for module in ("06_risk_knowledge_services", "07_decision_llm_engine", "08_recommendation_feedback"):
    sys.path.insert(0, str(ROOT / module))

from decision_engine.decision_agent import DecisionAgent
from recommendation_feedback.feedback_store import Feedback, FeedbackStore
from risk_knowledge.risk_model import LocalRiskModel


class DecisionFlowTest(unittest.TestCase):
    def test_normal_conditions_return_travel_normally(self):
        context = {"weather": "Normal conditions", "disaster": "No active alert", "transport": "Running normally"}
        risk = LocalRiskModel().assess(context, evidence_count=2).as_dict()
        result = DecisionAgent().decide(risk, {"alternative_available": False}).as_dict()
        self.assertEqual(result["action"], "travel_normally")

    def test_critical_restriction_returns_avoid_travel(self):
        context = {"weather": "Heavy snow", "disaster": "Emergency evacuation warning", "transport": "Line cancelled"}
        risk = LocalRiskModel().assess(context, evidence_count=2).as_dict()
        result = DecisionAgent().decide(risk, {"alternative_available": False}).as_dict()
        self.assertEqual(result["action"], "avoid_travel")
        self.assertTrue(result["emergency"])

    def test_feedback_is_stored_without_personal_data(self):
        record = FeedbackStore().submit(Feedback(recommendation_id="test-id", useful=True))
        self.assertEqual(record["recommendation_id"], "test-id")
        self.assertTrue(record["feedback_id"])


if __name__ == "__main__":
    unittest.main()
