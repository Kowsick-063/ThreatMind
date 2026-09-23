from __future__ import annotations

from copy import deepcopy
from typing import Any

import joblib
import numpy as np
import torch

from counterfactual.interventions import InterventionSpec, build_intervention
from counterfactual.scoring import score_risk
from counterfactual.state_transformer import apply_intervention
from models.world_model.create_sequences import FEATURES
from models.world_model.rollout import load_model


def _coerce_state_dict(state: dict[str, Any] | np.ndarray | list[Any]) -> dict[str, float]:
    if isinstance(state, dict):
        return {key: float(value) for key, value in state.items() if key in FEATURES}
    if isinstance(state, np.ndarray):
        return {feature: float(value) for feature, value in zip(FEATURES, state.tolist())}
    if isinstance(state, list) and state and isinstance(state[0], dict):
        return {key: float(value) for key, value in state[-1].items() if key in FEATURES}
    raise TypeError(f"Unsupported state format: {type(state)!r}")


def _coerce_sequence(sequence: np.ndarray | list[Any] | Any) -> np.ndarray:
    if isinstance(sequence, np.ndarray):
        return sequence.astype(np.float32)
    if isinstance(sequence, list):
        if not sequence:
            raise ValueError("Sequence cannot be empty.")
        if isinstance(sequence[0], dict):
            arr = []
            for item in sequence:
                arr.append([float(item.get(feature, 0.0)) for feature in FEATURES])
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(sequence, dtype=np.float32)
    raise TypeError(f"Unsupported sequence type: {type(sequence)!r}")


def _state_to_array(state: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(state.get(feature, 0.0)) for feature in FEATURES], dtype=np.float32)


def _derive_attack_probability(state: dict[str, Any]) -> float:
    traffic_pressure = (
        float(state.get("flow_count", 0.0)) / 200.0
        + float(state.get("total_packet_count", 0.0)) / 2000.0
        + float(state.get("total_byte_count", 0.0)) / 40000.0
        + float(state.get("mean_flow_packets_per_sec", 0.0)) / 10.0
        + float(state.get("mean_flow_bytes_per_sec", 0.0)) / 500.0
    )
    protocol_pressure = (
        float(state.get("syn_rate", 0.0)) * 2.5
        + float(state.get("ack_rate", 0.0)) * 1.25
        + float(state.get("rst_rate", 0.0)) * 1.5
    )
    probability = min(max((traffic_pressure + protocol_pressure) / 6.0, 0.0), 1.0)
    return round(probability, 6)


def _annotate_point(point: dict[str, Any]) -> dict[str, Any]:
    state = point.get("state") if isinstance(point.get("state"), dict) else {}
    probability = point.get("attack_probability")
    if probability is None:
        probability = _derive_attack_probability(state)
    probability = min(max(float(probability), 0.0), 1.0)
    category = "ATTACK" if probability >= 0.5 else "BENIGN"
    risk = score_risk(
        state=state,
        attack_probability=probability,
        category_confidence=probability,
    )
    point["attack_probability"] = round(probability, 6)
    point["attack_category"] = category
    point["category_confidence"] = round(probability, 6)
    point["risk"] = risk
    return point


def _predict_next(model, sequence: np.ndarray, scaler) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got {sequence.shape}")
    transformed = scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape).astype(np.float32)
    tensor = np.asarray(transformed, dtype=np.float32).reshape(1, -1, len(FEATURES))
    tensor = torch.from_numpy(tensor)
    with torch.no_grad():
        prediction = model(tensor)
    return np.asarray(prediction.numpy()[0], dtype=np.float32)


def generate_baseline_trajectory(
    initial_sequence,
    horizon: int = 12,
    forecasting_model=None,
    scaler=None,
):
    """Generate a deterministic baseline trajectory using the existing world model."""
    if forecasting_model is None:
        forecasting_model = load_model()
    if scaler is None:
        scaler = joblib.load("models/world_model/world_model_scaler.joblib")

    sequence = _coerce_sequence(initial_sequence)
    trajectory = []
    current_sequence = sequence.copy()

    for step in range(1, horizon + 1):
        next_state = _predict_next(forecasting_model, current_sequence, scaler)
        state = {feature: float(next_state[index]) for index, feature in enumerate(FEATURES)}
        trajectory.append(_annotate_point({
            "horizon": step,
            "state": state,
            "attack_probability": None,
            "attack_category": None,
            "category_confidence": None,
        }))
        current_sequence = np.vstack([current_sequence[1:], next_state.reshape(1, -1)])

    return trajectory


def simulate_counterfactual(
    initial_state,
    intervention,
    forecasting_model=None,
    scaler=None,
    horizon: int = 12,
):
    """Simulate a counterfactual trajectory by transforming the recent state and rolling out forward."""
    if not isinstance(intervention, InterventionSpec):
        intervention = build_intervention(intervention["type"], intervention.get("target", 0)) if isinstance(intervention, dict) else build_intervention(intervention)

    if forecasting_model is None:
        forecasting_model = load_model()
    if scaler is None:
        scaler = joblib.load("models/world_model/world_model_scaler.joblib")

    state_dict = _coerce_state_dict(initial_state)
    modified = apply_intervention(state_dict, intervention)

    sequence = np.asarray([
        [float(state_dict.get(feature, 0.0)) for feature in FEATURES]
        for _ in range(60)
    ], dtype=np.float32)
    sequence[-1] = _state_to_array(modified)

    baseline = generate_baseline_trajectory(sequence, horizon, forecasting_model, scaler)
    counterfactual = []
    current_sequence = sequence.copy()
    for step in range(1, horizon + 1):
        transformed_last = apply_intervention(
            {feature: float(current_sequence[-1, index]) for index, feature in enumerate(FEATURES)},
            intervention,
        )
        current_sequence = np.vstack([current_sequence[1:], _state_to_array(transformed_last)])
        next_state = _predict_next(forecasting_model, current_sequence, scaler)
        counterfactual.append(_annotate_point({
            "horizon": step,
            "state": {feature: float(next_state[index]) for index, feature in enumerate(FEATURES)},
            "attack_probability": None,
            "attack_category": None,
            "category_confidence": None,
        }))

    return {
        "baseline": baseline,
        "counterfactual": counterfactual,
        "intervention": {
            "type": intervention.type,
            "target": intervention.target,
            "operational_cost": intervention.operational_cost,
            "service_disruption_cost": intervention.service_disruption_cost,
        },
    }


__all__ = [
    "generate_baseline_trajectory",
    "simulate_counterfactual",
]
