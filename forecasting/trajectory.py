"""Multi-step attack trajectory generation pipeline for ThreatMind.

Produces structured H1–H12 attack trajectories by fusing:
  - Production LSTM world model (absolute state rollout)
  - Delta-LSTM world model (change signal rollout)
  - AttackProbabilityModel + 12 horizon-specific Platt calibrators
  - AttackCategoryEstimator (with volumetric DDoS support guard)
  - MITRE ATT&CK technique evidence mapper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from counterfactual.rollout import _load_annotation_models, get_historical_sequence
from forecasting.trajectory_explanation import generate_trajectory_explanation
from forecasting.trajectory_fusion import fuse_trajectory_step
from models.world_model.create_sequences import FEATURES, SEQUENCE_LENGTH
from models.world_model.delta_rollout import load_model as load_delta_model, recursive_delta_rollout
from models.world_model.rollout import load_model as load_lstm_model


_SCALER_PATH = Path("models/world_model/world_model_scaler.joblib")
_DELTA_SCALER_PATH = Path("models/world_model/delta_world_model_scaler.joblib")
HORIZON_STEPS = 12


def generate_attack_trajectory(
    current_timestamp: float | int | str | None = None,
    current_state: dict[str, float] | np.ndarray | None = None,
    historical_sequence: np.ndarray | list[Any] | None = None,
    horizon: int = HORIZON_STEPS,
) -> dict[str, Any]:
    """Generate a structured H1-H12 attack trajectory fusing ThreatMind models.

    Parameters
    ----------
    current_timestamp:
        Timestamp of the current state. Used to fetch contiguous 60-state history
        if historical_sequence is not directly supplied.
    current_state:
        Optional observed network state at t=0. Defaults to the final row of
        the historical sequence.
    historical_sequence:
        Optional contiguous 60-step historical sequence (shape (60, 26)).
        If omitted, fetched via get_historical_sequence(current_timestamp).
    horizon:
        Number of forecast steps ahead (defaults to 12).

    Returns
    -------
    dict containing:
      - current_timestamp: float | int | str | None
      - current_state: dict[str, float]
      - trajectory: list[dict] (H1 through H12 trajectory points)
      - trajectory_summary: dict (peak probability, trend, category transition, risk horizons)
      - mitre_evidence: list[dict]
      - explanation: dict
      - uncertainties: list[str]
    """
    if horizon < 1 or horizon > 12:
        raise ValueError(f"horizon must be between 1 and 12, got {horizon}.")

    # ------------------------------------------------------------------
    # 1. Historical sequence retrieval and validation
    # ------------------------------------------------------------------
    if historical_sequence is None:
        if current_timestamp is None:
            raise ValueError(
                "Either current_timestamp or historical_sequence must be provided."
            )
        # Raises ValueError if missing or if non-5s gaps / insufficient history exist.
        # Strictly avoids silent fallback to repeated states.
        seq_array = get_historical_sequence(current_timestamp, sequence_length=SEQUENCE_LENGTH)
    else:
        seq_array = np.asarray(historical_sequence, dtype=np.float32)
        if seq_array.ndim == 3 and seq_array.shape[0] == 1:
            seq_array = seq_array[0]
        if seq_array.shape != (SEQUENCE_LENGTH, len(FEATURES)):
            raise ValueError(
                f"historical_sequence must have shape ({SEQUENCE_LENGTH}, {len(FEATURES)}), "
                f"got {seq_array.shape}."
            )
        if not np.all(np.isfinite(seq_array)):
            raise ValueError("historical_sequence contains non-finite values.")

    # ------------------------------------------------------------------
    # 2. Coerce current state
    # ------------------------------------------------------------------
    if current_state is not None:
        if isinstance(current_state, dict):
            state_dict = {f: float(current_state.get(f, 0.0)) for f in FEATURES}
        elif isinstance(current_state, np.ndarray):
            state_dict = {f: float(val) for f, val in zip(FEATURES, current_state.tolist())}
        else:
            raise TypeError(f"Unsupported current_state format: {type(current_state)!r}")
    else:
        state_dict = {
            f: float(seq_array[-1, i]) for i, f in enumerate(FEATURES)
        }

    # ------------------------------------------------------------------
    # 3. Load model artifacts and scalers
    # ------------------------------------------------------------------
    lstm_model = load_lstm_model()
    scaler = joblib.load(_SCALER_PATH)
    probability_model, category_estimator, calibrators = _load_annotation_models()

    delta_available = True
    delta_error: str | None = None
    delta_model = None
    delta_scaler = None
    try:
        delta_model = load_delta_model()
        delta_scaler = joblib.load(_DELTA_SCALER_PATH)
    except Exception as exc:  # pragma: no cover
        delta_available = False
        delta_error = str(exc)

    # ------------------------------------------------------------------
    # 4. Production LSTM rollout (Primary absolute state world model)
    # ------------------------------------------------------------------
    scaled_history = scaler.transform(seq_array).astype(np.float32)
    seq_tensor = torch.from_numpy(scaled_history.reshape(1, SEQUENCE_LENGTH, len(FEATURES)))

    production_scaled_states: list[np.ndarray] = []
    production_real_states: list[dict[str, float]] = []

    with torch.no_grad():
        for _ in range(horizon):
            next_scaled = lstm_model(seq_tensor)  # (1, 26)
            seq_tensor = torch.cat(
                [seq_tensor[:, 1:, :], next_scaled.unsqueeze(1)],
                dim=1,
            )
            scaled_state = next_scaled.numpy()[0]
            real_state = scaler.inverse_transform(scaled_state.reshape(1, -1))[0]

            production_scaled_states.append(scaled_state)
            production_real_states.append(
                {f: float(real_state[i]) for i, f in enumerate(FEATURES)}
            )

    # ------------------------------------------------------------------
    # 5. Delta-LSTM rollout (Secondary change signal)
    # ------------------------------------------------------------------
    delta_rollout_states: np.ndarray | None = None
    if delta_available:
        try:
            delta_rollout_states = recursive_delta_rollout(
                initial_sequences=seq_array,
                horizon=horizon,
                model=delta_model,
                state_scaler=scaler,
                delta_scaler=delta_scaler,
            )  # (1, horizon, 26)
        except Exception as exc:  # pragma: no cover
            delta_available = False
            delta_error = str(exc)

    # ------------------------------------------------------------------
    # 6. Fuse horizons H1..H12
    # ------------------------------------------------------------------
    trajectory_points: list[dict[str, Any]] = []

    for step in range(1, horizon + 1):
        # Timestamp calculation
        if current_timestamp is not None:
            try:
                numeric_ts = float(current_timestamp)
                step_timestamp = numeric_ts + (step * 5.0)
            except (ValueError, TypeError):
                step_timestamp = f"{current_timestamp}+H{step}"
        else:
            step_timestamp = f"t+{step * 5}s"

        prod_scaled = production_scaled_states[step - 1]
        prod_real = production_real_states[step - 1]

        # Extract delta signals
        if delta_available and delta_rollout_states is not None:
            if step == 1:
                delta_change_arr = delta_rollout_states[0, 0] - seq_array[-1]
            else:
                delta_change_arr = (
                    delta_rollout_states[0, step - 1]
                    - delta_rollout_states[0, step - 2]
                )
            delta_change_dict = {
                f: float(delta_change_arr[i]) for i, f in enumerate(FEATURES)
            }
            delta_reconstructed_dict = {
                f: float(delta_rollout_states[0, step - 1, i])
                for i, f in enumerate(FEATURES)
            }
        else:
            delta_change_dict = {}
            delta_reconstructed_dict = None

        point = fuse_trajectory_step(
            horizon=step,
            timestamp=step_timestamp,
            raw_lstm_state_scaled=prod_scaled,
            production_state_real=prod_real,
            delta_change_real=delta_change_dict,
            delta_reconstructed_real=delta_reconstructed_dict,
            probability_model=probability_model,
            calibrator=calibrators[step],
            category_estimator=category_estimator,
            delta_available=delta_available,
            delta_error=delta_error,
        )
        trajectory_points.append(point)

    # ------------------------------------------------------------------
    # 7. Trajectory summary
    # ------------------------------------------------------------------
    probs = [p["predicted_attack_probability"] for p in trajectory_points]
    peak_prob = max(probs)
    peak_horizon = probs.index(peak_prob) + 1
    init_prob = probs[0]
    final_prob = probs[-1]

    diff = final_prob - init_prob
    if diff > 0.05:
        prob_trend = "increasing"
    elif diff < -0.05:
        prob_trend = "decreasing"
    else:
        prob_trend = "stable"

    all_cats = [p["attack_category"] for p in trajectory_points]
    category_transitions: list[str] = []
    for c in all_cats:
        if not category_transitions or category_transitions[-1] != c:
            category_transitions.append(c)

    risk_horizons = [
        p["horizon"]
        for p in trajectory_points
        if p["predicted_attack_probability"] >= 0.5
    ]

    trajectory_summary = {
        "peak_attack_probability": round(peak_prob, 6),
        "peak_horizon": peak_horizon,
        "initial_attack_probability": round(init_prob, 6),
        "final_attack_probability": round(final_prob, 6),
        "probability_trend": prob_trend,
        "category_transition": category_transitions,
        "risk_horizons": risk_horizons,
    }

    # ------------------------------------------------------------------
    # 8. Explainability synthesis
    # ------------------------------------------------------------------
    explanation = generate_trajectory_explanation(
        current_state=state_dict,
        trajectory=trajectory_points,
        trajectory_summary=trajectory_summary,
        current_timestamp=current_timestamp,
    )

    return {
        "current_timestamp": current_timestamp,
        "current_state": state_dict,
        "trajectory": trajectory_points,
        "trajectory_summary": trajectory_summary,
        "mitre_evidence": explanation["mitre_evidence"],
        "explanation": explanation,
        "uncertainties": explanation["uncertainties"],
    }


__all__ = [
    "HORIZON_STEPS",
    "generate_attack_trajectory",
]
