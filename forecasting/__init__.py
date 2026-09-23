"""Forecasting package for ThreatMind.

Provides multi-step trajectory generation, multi-model forecast fusion,
and explainability.
"""

from forecasting.trajectory import HORIZON_STEPS, generate_attack_trajectory
from forecasting.trajectory_explanation import generate_trajectory_explanation
from forecasting.trajectory_fusion import (
    SUPPORTED_ESTIMATOR_CLASSES,
    detect_ddos_pattern,
    fuse_trajectory_step,
)

__all__ = [
    "HORIZON_STEPS",
    "SUPPORTED_ESTIMATOR_CLASSES",
    "detect_ddos_pattern",
    "fuse_trajectory_step",
    "generate_attack_trajectory",
    "generate_trajectory_explanation",
]
