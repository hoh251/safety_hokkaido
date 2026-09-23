"""Conservative, explainable local risk scoring for the decision layer."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RiskAssessment:
    level: str
    score: int
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"level": self.level, "score": self.score, "reasons": self.reasons}


class LocalRiskModel:
    """Rule-based baseline; it is explicit until a validated ML model replaces it."""

    CRITICAL_TERMS = ("tsunami", "evacuate", "emergency", "closure", "cancelled")
    HIGH_TERMS = ("earthquake", "warning", "blizzard", "storm", "flood", "wildfire", "heavy snow")
    MODERATE_TERMS = ("delay", "snow", "wind", "unavailable")

    def assess(self, travel_context: Dict[str, object], evidence_count: int = 0) -> RiskAssessment:
        source_text = " ".join(
            str(travel_context.get(source, "")).lower()
            for source in ("weather", "disaster", "transport")
        )
        score = 0
        reasons: List[str] = []

        if any(term in source_text for term in self.CRITICAL_TERMS):
            score += 80
            reasons.append("A critical safety or transport restriction was reported.")
        if any(term in source_text for term in self.HIGH_TERMS):
            score += 45
            reasons.append("Live sources reported a potentially hazardous condition.")
        if any(term in source_text for term in self.MODERATE_TERMS):
            score += 20
            reasons.append("Live data is degraded or reports a possible delay/weather impact.")
        if evidence_count == 0:
            score += 10
            reasons.append("No matching local safety document was retrieved; confidence is limited.")

        score = min(score, 100)
        level = "high" if score >= 70 else "medium" if score >= 35 else "low"
        return RiskAssessment(level=level, score=score, reasons=reasons or ["No high-risk signal was found in the available sources."])
