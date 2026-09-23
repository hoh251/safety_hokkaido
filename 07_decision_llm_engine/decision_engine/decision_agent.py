"""Deterministic safety decision before the LLM writes an explanation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Recommendation:
    action: str
    risk_level: str
    confidence: str
    reasons: List[str]
    emergency: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "emergency": self.emergency,
        }


class DecisionAgent:
    def decide(self, risk: Dict[str, object], route: Dict[str, object]) -> Recommendation:
        score = int(risk["score"])
        reasons = list(risk["reasons"])
        if score >= 80:
            return Recommendation("avoid_travel", str(risk["level"]), "medium", reasons, emergency=True)
        if score >= 60:
            return Recommendation("avoid_travel", str(risk["level"]), "medium", reasons)
        if score >= 35 and route.get("alternative_available"):
            return Recommendation("change_route", str(risk["level"]), "low", reasons)
        if score >= 35:
            return Recommendation("delay_travel", str(risk["level"]), "low", reasons)
        return Recommendation("travel_normally", str(risk["level"]), "medium", reasons)
