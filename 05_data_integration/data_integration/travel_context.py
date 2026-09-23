"""Combines live-source responses and traveler input into one auditable snapshot."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from external_data.tools import (
    check_train_status,
    get_disaster_warnings,
    get_real_time_weather,
)


@dataclass
class TravelRequest:
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_time: Optional[str] = None
    transport_mode: Optional[str] = None


@dataclass
class TravelContext:
    traveler: TravelRequest
    weather: str = "Not requested."
    disaster: str = "Not requested."
    transport: str = "Not requested."
    route: Dict[str, object] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class TravelContextBuilder:
    """Keeps provider calls outside the LLM and records every returned source."""

    def __init__(
        self,
        weather: Callable[[str], str] = get_real_time_weather,
        disaster: Callable[[str], str] = get_disaster_warnings,
        transport: Callable[[str], str] = check_train_status,
    ):
        self.weather = weather
        self.disaster = disaster
        self.transport = transport

    def build(self, traveler: TravelRequest, enabled: Dict[str, bool]) -> TravelContext:
        destination = traveler.destination or traveler.origin or "Hokkaido"
        mode = traveler.transport_mode or "All"
        context = TravelContext(traveler=traveler)

        if enabled.get("weather"):
            context.weather = self.weather(destination)
        if enabled.get("disaster"):
            context.disaster = self.disaster("Hokkaido")
        if enabled.get("train"):
            context.transport = self.transport(mode)

        context.route = {
            "origin": traveler.origin,
            "destination": traveler.destination,
            "mode": traveler.transport_mode,
            "status": "route details unavailable" if not traveler.origin or not traveler.destination else "route requested",
            "alternative_available": False,
        }
        return context
