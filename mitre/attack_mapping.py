from __future__ import annotations

from typing import Any


MITRE_MAP = {
    "BENIGN": [],
    "Bot": [
        {
            "attack_category": "Bot",
            "technique_id": "T1059",
            "technique_name": "Command and Scripting Interpreter",
            "confidence": 0.45,
            "evidence": "Observed bot-like activity in the temporal state representation.",
        }
    ],
    "PortScan": [
        {
            "attack_category": "PortScan",
            "technique_id": "T1046",
            "technique_name": "Network Service Discovery",
            "confidence": 0.7,
            "evidence": "Port enumeration patterns are consistent with service discovery behavior.",
        }
    ],
    "DDoS": [
        {
            "attack_category": "DDoS",
            "technique_id": "T1498",
            "technique_name": "Network Denial of Service",
            "confidence": 0.8,
            "evidence": "High-volume traffic and synchronized service degradation patterns are consistent with denial of service.",
        }
    ],
}


def map_attack_category(category: str | None, fallback: str | None = None) -> list[dict[str, Any]]:
    """Map only categories actually observed in the training/forecasting pipeline to MITRE techniques."""
    if category is None:
        category = fallback
    if category is None:
        return []
    return MITRE_MAP.get(str(category), [])


__all__ = ["MITRE_MAP", "map_attack_category"]
