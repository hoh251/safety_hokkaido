"""Privacy-minimized in-memory feedback loop for local development."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4


@dataclass
class Feedback:
    recommendation_id: str
    useful: bool
    comment: Optional[str] = None
    feedback_id: str = ""
    submitted_at: str = ""


class FeedbackStore:
    def __init__(self):
        self._items: List[Feedback] = []
        self._lock = Lock()

    def submit(self, feedback: Feedback) -> Dict[str, object]:
        feedback.feedback_id = str(uuid4())
        feedback.submitted_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._items.append(feedback)
        return asdict(feedback)

    def summary(self) -> Dict[str, int]:
        with self._lock:
            return {"total": len(self._items), "useful": sum(item.useful for item in self._items)}
