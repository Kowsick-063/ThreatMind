"""MITRE ATT&CK mapping and explainability for ThreatMind.

This package provides:
- attack_techniques: Static technique registry grounded in CIC-IDS2017 evidence.
- technique_mapper:  Evidence-based technique mapper.
- explanation:       Structured explanation generator.
"""

from mitre.attack_techniques import TECHNIQUE_REGISTRY, AttackTechnique
from mitre.technique_mapper import map_network_behavior_to_techniques
from mitre.explanation import generate_attack_explanation

__all__ = [
    "TECHNIQUE_REGISTRY",
    "AttackTechnique",
    "map_network_behavior_to_techniques",
    "generate_attack_explanation",
]
