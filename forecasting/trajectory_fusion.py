"""Transparent deterministic forecast fusion layer for ThreatMind.

Fuses:
  - Absolute state forecast       → production LSTM (models/world_model/lstm_world_model.pt)
  - Change signal (delta)         → Delta-LSTM (models/world_model/delta_lstm_world_model.pt)
  - Attack probability            → AttackProbabilityModel + horizon-specific Platt calibrator
  - Attack category               → AttackCategoryEstimator (with explicit DDoS boundary guard)
  - MITRE ATT&CK techniques       → mitre.technique_mapper (evidence-based, decoupled from probability)

Maintains full provenance for every prediction signal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mitre.technique_mapper import map_network_behavior_to_techniques
from models.world_model.attack_category_estimation import AttackCategoryEstimator
from models.world_model.attack_probability_model import AttackProbabilityModel
from models.world_model.create_sequences import FEATURES


SUPPORTED_ESTIMATOR_CLASSES = ["BENIGN", "Bot", "PortScan"]


def detect_ddos_pattern(state: dict[str, float]) -> bool:
    """Detect volumetric DoS/DDoS traffic signatures.

    Uses conditions aligned with MITRE T1498 (packets_per_second > 1000)
    and heavy volumetric flow/packet/byte signatures.
    """
    pps = float(state.get("packets_per_second", 0.0))
    packet_count = float(state.get("packet_count", 0.0))
    syn_count = float(state.get("syn_count", 0.0))
    byte_count = float(state.get("byte_count", 0.0))

    if pps > 1000.0:
        return True
    if packet_count > 50000.0 or syn_count > 50000.0 or byte_count > 50_000_000.0:
        return True
    return False


def fuse_trajectory_step(
    horizon: int,
    timestamp: float | str,
    raw_lstm_state_scaled: np.ndarray,
    production_state_real: dict[str, float],
    delta_change_real: dict[str, float],
    delta_reconstructed_real: dict[str, float] | None,
    probability_model: AttackProbabilityModel,
    calibrator: Any,
    category_estimator: AttackCategoryEstimator,
    delta_available: bool = True,
    delta_error: str | None = None,
) -> dict[str, Any]:
    """Fuse multi-model outputs for a single trajectory horizon step.

    Parameters
    ----------
    horizon:
        1-indexed forecast horizon (1..12).
    timestamp:
        Timestamp for this horizon point.
    raw_lstm_state_scaled:
        26-dim array in StandardScaler-normalized space (raw production LSTM output).
    production_state_real:
        26-dim dictionary of real/unscaled network state features from production LSTM.
    delta_change_real:
        26-dim dictionary of unscaled feature deltas from Delta-LSTM.
    delta_reconstructed_real:
        26-dim dictionary of reconstructed state from Delta-LSTM rollout.
    probability_model:
        Trained AttackProbabilityModel.
    calibrator:
        Horizon-specific Platt calibrator for this horizon.
    category_estimator:
        Trained AttackCategoryEstimator.
    delta_available:
        Whether Delta-LSTM rollout was successfully executed.
    delta_error:
        Optional error string if Delta-LSTM execution failed.

    Returns
    -------
    dict matching the ThreatMind H1-H12 trajectory point schema.
    """
    uncertainties: list[str] = []

    # ------------------------------------------------------------------
    # 1. Attack probability: raw model -> logit -> horizon Platt calibrator
    # ------------------------------------------------------------------
    row = raw_lstm_state_scaled.reshape(1, -1)
    raw_prob = float(probability_model.predict_probability(row)[0])
    raw_prob = min(max(raw_prob, 0.0), 1.0)

    logit_val = math.log(max(raw_prob, 1e-7) / max(1.0 - raw_prob, 1e-7))
    calibrated_prob = float(calibrator.predict_proba(np.array([[logit_val]]))[0, 1])
    calibrated_prob = min(max(calibrated_prob, 0.0), 1.0)
    prob_source = f"AttackProbabilityModel + PlattCalibrator(horizon_{horizon:02d})"

    # ------------------------------------------------------------------
    # 2. Attack category & confidence (with DDoS support guard)
    # ------------------------------------------------------------------
    is_ddos = detect_ddos_pattern(production_state_real)
    if is_ddos:
        attack_category = "DDoS"
        category_confidence = 0.0
        uncertainties.append(
            "category_support_limited: volumetric flood detected consistent with DDoS, "
            "but AttackCategoryEstimator classes are limited to BENIGN, Bot, PortScan"
        )
        cat_source = "volumetric_pattern_detector (AttackCategoryEstimator unsupported class)"
    else:
        attack_category = str(category_estimator.predict(row)[0])
        class_proba = category_estimator.predict_proba(row)[0]
        confidence = float(np.nanmax(class_proba))
        category_confidence = min(max(confidence, 0.0), 1.0)
        cat_source = "AttackCategoryEstimator"

    # ------------------------------------------------------------------
    # 3. State forecasts with explicit provenance
    # ------------------------------------------------------------------
    absolute_forecast = {
        "source": "production_lstm",
        "model_file": "models/world_model/lstm_world_model.pt",
        "role": "primary_production_world_model",
        "state_features": production_state_real,
    }

    delta_forecast: dict[str, Any] = {
        "source": "delta_lstm",
        "model_file": "models/world_model/delta_lstm_world_model.pt",
        "role": "secondary_change_signal",
        "available": delta_available,
        "change_features": delta_change_real,
        "reconstructed_state": delta_reconstructed_real,
    }
    if not delta_available:
        delta_forecast["error"] = delta_error
        uncertainties.append(f"delta_lstm_unavailable: {delta_error}")

    # ------------------------------------------------------------------
    # 4. MITRE technique mapping (evidence-based; NOT driven by attack_prob)
    # ------------------------------------------------------------------
    technique_result = map_network_behavior_to_techniques(
        state=production_state_real,
        attack_category=attack_category,
        attack_probability=calibrated_prob,
    )
    mitre_techniques = technique_result.get("techniques", [])

    return {
        "timestamp": timestamp,
        "horizon": horizon,
        "predicted_attack_probability": round(calibrated_prob, 6),
        "attack_category": attack_category,
        "category_confidence": round(category_confidence, 6),
        "state_features": production_state_real,
        "change_features": delta_change_real,
        "absolute_forecast": absolute_forecast,
        "delta_forecast": delta_forecast,
        "mitre_techniques": mitre_techniques,
        "uncertainties": uncertainties,
        "sources": {
            "predicted_attack_probability": prob_source,
            "attack_category": cat_source,
            "absolute_forecast": "production_lstm",
            "delta_forecast": "delta_lstm",
        },
    }


__all__ = [
    "SUPPORTED_ESTIMATOR_CLASSES",
    "detect_ddos_pattern",
    "fuse_trajectory_step",
]
