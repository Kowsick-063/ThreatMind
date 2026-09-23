from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from counterfactual.interventions import InterventionSpec, build_intervention
from counterfactual.scoring import score_risk
from counterfactual.state_transformer import apply_intervention
from models.world_model.attack_category_estimation import AttackCategoryEstimator
from models.world_model.attack_probability_model import AttackProbabilityModel
from models.world_model.create_sequences import (
    FEATURES,
    INPUT_FILE,
    SEQUENCE_LENGTH,
)
from models.world_model.rollout import load_model


_CALIBRATOR_DIR = Path("models/world_model/calibrators")
_APM_PATH = Path("models/world_model/attack_probability_model.joblib")
_ACE_PATH = Path("models/world_model/attack_category_estimation.joblib")
_SCALER_PATH = Path("models/world_model/world_model_scaler.joblib")
_HISTORICAL_STATES_PATH = INPUT_FILE

# ---------------------------------------------------------------------------
# Module-level artifact cache — loaded once per process, reused across calls.
# ---------------------------------------------------------------------------
_probability_model: AttackProbabilityModel | None = None
_category_estimator: AttackCategoryEstimator | None = None
_calibrators: dict[int, Any] | None = None


def _load_annotation_models() -> tuple[AttackProbabilityModel, AttackCategoryEstimator, dict]:
    """Load and cache AttackProbabilityModel, AttackCategoryEstimator, and all
    12 horizon-specific Platt calibrators.

    Both the probability model and the category estimator were trained on
    StandardScaler-normalized features.  Pass the raw LSTM output (scaled
    space) directly — do NOT inverse-transform before calling these models.

    Raises FileNotFoundError with a clear message if any artifact is missing.
    """
    global _probability_model, _category_estimator, _calibrators
    if (
        _probability_model is not None
        and _category_estimator is not None
        and _calibrators is not None
    ):
        return _probability_model, _category_estimator, _calibrators

    if not _APM_PATH.exists():
        raise FileNotFoundError(
            f"AttackProbabilityModel artifact not found: {_APM_PATH}. "
            "Re-run the training pipeline before starting the server."
        )
    if not _ACE_PATH.exists():
        raise FileNotFoundError(
            f"AttackCategoryEstimator artifact not found: {_ACE_PATH}. "
            "Re-run the training pipeline before starting the server."
        )

    _probability_model = AttackProbabilityModel.load(_APM_PATH)
    _category_estimator = AttackCategoryEstimator.load(_ACE_PATH)

    _calibrators = {}
    for horizon in range(1, 13):
        path = _CALIBRATOR_DIR / f"horizon_{horizon:02d}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Platt calibrator not found: {path}. "
                "Re-run calibrate_probability_horizons before starting the server."
            )
        _calibrators[horizon] = joblib.load(path)

    return _probability_model, _category_estimator, _calibrators


# ---------------------------------------------------------------------------
# State coercion helpers
# ---------------------------------------------------------------------------

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
            arr = [
                [float(item.get(feature, 0.0)) for feature in FEATURES]
                for item in sequence
            ]
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(sequence, dtype=np.float32)
    raise TypeError(f"Unsupported sequence type: {type(sequence)!r}")


def _state_to_array(state: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(state.get(feature, 0.0)) for feature in FEATURES],
        dtype=np.float32,
    )


def get_historical_sequence(
    current_timestamp: float | int | str | None,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Return contiguous raw feature history ending at ``current_timestamp``.

    The returned rows are ordered oldest-to-newest and contain only the LSTM's
    26 feature columns. A missing timestamp or any non-five-second interval in
    the requested history raises an explicit error instead of bridging a gap.
    """
    if current_timestamp is None:
        raise ValueError(
            "current_timestamp is required to retrieve historical state history."
        )
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    try:
        requested_timestamp = float(current_timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid current_timestamp: {current_timestamp!r}."
        ) from error

    if not _HISTORICAL_STATES_PATH.exists():
        raise FileNotFoundError(
            f"Historical state dataset not found: {_HISTORICAL_STATES_PATH}."
        )

    data = pd.read_csv(_HISTORICAL_STATES_PATH)
    required_columns = ["timestamp", *FEATURES]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Historical state dataset is missing columns: {missing_columns}"
        )

    timestamps = pd.to_numeric(data["timestamp"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Historical state timestamps contain non-finite values.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Historical state timestamps must be strictly increasing.")

    matching_indices = np.flatnonzero(timestamps == requested_timestamp)
    if len(matching_indices) == 0:
        raise ValueError(
            f"No historical state exists at current_timestamp={requested_timestamp}."
        )

    end_index = int(matching_indices[0])
    start_index = end_index - sequence_length + 1
    if start_index < 0:
        raise ValueError(
            "Contiguous historical sequence unavailable: "
            f"current_timestamp={requested_timestamp} has only "
            f"{end_index + 1} available states, requires {sequence_length}."
        )

    selected_timestamps = timestamps[start_index:end_index + 1]
    gaps = np.diff(selected_timestamps)
    invalid_gap = np.flatnonzero(gaps != 5.0)
    if len(invalid_gap) > 0:
        gap_index = int(invalid_gap[0])
        raise ValueError(
            "Contiguous historical sequence unavailable: temporal gap "
            f"between {selected_timestamps[gap_index]} and "
            f"{selected_timestamps[gap_index + 1]} is "
            f"{gaps[gap_index]} seconds; expected 5.0 seconds."
        )

    values = data.loc[start_index:end_index, FEATURES].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Historical state features contain non-finite values.")
    return values


# ---------------------------------------------------------------------------
# LSTM step — returns raw scaled-space output (26-dim)
# ---------------------------------------------------------------------------

def _predict_next(model, sequence: np.ndarray, scaler) -> np.ndarray:
    """Scale *sequence*, feed to LSTM, return raw scaled output (26-dim).

    The returned array is in StandardScaler-normalized space.  It is passed
    directly to AttackProbabilityModel and AttackCategoryEstimator, which
    were both trained on scaled features.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got {sequence.shape}")
    transformed = (
        scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
        .reshape(sequence.shape)
        .astype(np.float32)
    )
    tensor = torch.from_numpy(
        transformed.reshape(1, -1, len(FEATURES)).astype(np.float32)
    )
    with torch.no_grad():
        prediction = model(tensor)
    return np.asarray(prediction.numpy()[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# ML annotation — replaces the old heuristic _derive_attack_probability
# ---------------------------------------------------------------------------

def _annotate_point(
    point: dict[str, Any],
    scaled_state: np.ndarray,
    horizon: int,
    probability_model: AttackProbabilityModel,
    calibrators: dict,
    category_estimator: AttackCategoryEstimator,
) -> dict[str, Any]:
    """Annotate one trajectory point using the full trained ML pipeline.

    Parameters
    ----------
    point:
        Mutable trajectory dict.  Must already contain ``state`` and
        ``horizon``.  This function populates ``attack_probability``,
        ``attack_category``, ``category_confidence``, and ``risk``.
    scaled_state:
        26-dim float32 array in StandardScaler-normalized space (raw LSTM
        output).  Passed directly to AttackProbabilityModel and
        AttackCategoryEstimator — both were trained on scaled features.
    horizon:
        1-indexed rollout step (1..12).  Selects the correct Platt calibrator.
    """
    row = scaled_state.reshape(1, -1)

    # 1. Raw binary attack probability from the trained logistic classifier.
    raw_prob = float(probability_model.predict_probability(row)[0])
    raw_prob = min(max(raw_prob, 0.0), 1.0)

    # 2. Horizon-specific Platt calibration.
    #    The calibrator expects logit(raw_prob) as its single input feature.
    logit_val = math.log(max(raw_prob, 1e-7) / max(1.0 - raw_prob, 1e-7))
    calibrated_prob = float(
        calibrators[horizon].predict_proba(np.array([[logit_val]]))[0, 1]
    )
    calibrated_prob = min(max(calibrated_prob, 0.0), 1.0)

    # 3. Attack category from the trained nearest-centroid estimator.
    #    Supported classes: BENIGN, Bot, PortScan.
    category = str(category_estimator.predict(row)[0])

    # 4. Category confidence = max class probability.
    #    nanmax guards against NaN if softmax denominator underflows.
    class_proba = category_estimator.predict_proba(row)[0]
    confidence = float(np.nanmax(class_proba))
    confidence = min(max(confidence, 0.0), 1.0)

    # 5. Risk score using calibrated probability and category confidence.
    state = point.get("state") if isinstance(point.get("state"), dict) else {}
    risk = score_risk(
        state=state,
        attack_probability=calibrated_prob,
        category_confidence=confidence,
    )

    point["attack_probability"] = round(calibrated_prob, 6)
    point["attack_category"] = category
    point["category_confidence"] = round(confidence, 6)
    point["risk"] = risk
    return point


# ---------------------------------------------------------------------------
# Public trajectory generators
# ---------------------------------------------------------------------------

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
        scaler = joblib.load(str(_SCALER_PATH))

    probability_model, category_estimator, calibrators = _load_annotation_models()

    sequence = _coerce_sequence(initial_sequence)
    trajectory = []
    current_sequence = sequence.copy()

    for step in range(1, horizon + 1):
        # _predict_next returns 26-dim array in StandardScaler-normalized space.
        next_state_scaled = _predict_next(forecasting_model, current_sequence, scaler)
        state = {
            feature: float(next_state_scaled[index])
            for index, feature in enumerate(FEATURES)
        }
        point = {
            "horizon": step,
            "state": state,
            "attack_probability": None,
            "attack_category": None,
            "category_confidence": None,
        }
        trajectory.append(
            _annotate_point(
                point,
                scaled_state=next_state_scaled,
                horizon=step,
                probability_model=probability_model,
                calibrators=calibrators,
                category_estimator=category_estimator,
            )
        )
        # Slide the window forward with the scaled LSTM output.
        current_sequence = np.vstack(
            [current_sequence[1:], next_state_scaled.reshape(1, -1)]
        )

    return trajectory


def simulate_counterfactual(
    initial_state,
    intervention,
    forecasting_model=None,
    scaler=None,
    horizon: int = 12,
    current_timestamp: float | int | str | None = None,
):
    """Simulate a counterfactual trajectory by transforming the recent state and
    rolling out forward.

    Both baseline and counterfactual legs are annotated with the same trained ML
    pipeline so comparator.py receives equivalent ML-generated predictions.
    """
    if not isinstance(intervention, InterventionSpec):
        if isinstance(intervention, dict):
            intervention = build_intervention(
                intervention["type"], intervention.get("target", 0)
            )
        else:
            intervention = build_intervention(intervention)

    state_dict = _coerce_state_dict(initial_state)
    modified = apply_intervention(state_dict, intervention)

    if current_timestamp is None:
        # Backward-compatible mode for callers that have no temporal identity.
        sequence = np.asarray(
            [
                [float(state_dict.get(feature, 0.0)) for feature in FEATURES]
                for _ in range(SEQUENCE_LENGTH)
            ],
            dtype=np.float32,
        )
    else:
        sequence = get_historical_sequence(current_timestamp)

    # Preserve the existing intervention semantics: both rollout legs start
    # from the same history, with the intervention applied to its newest row.
    sequence[-1] = _state_to_array(modified)

    if forecasting_model is None:
        forecasting_model = load_model()
    if scaler is None:
        scaler = joblib.load(str(_SCALER_PATH))

    probability_model, category_estimator, calibrators = _load_annotation_models()

    # Baseline: no per-step intervention during rollout.
    baseline = generate_baseline_trajectory(
        sequence, horizon, forecasting_model, scaler
    )

    # Counterfactual: intervention re-applied at every rollout step.
    counterfactual = []
    current_sequence = sequence.copy()

    for step in range(1, horizon + 1):
        transformed_last = apply_intervention(
            {
                feature: float(current_sequence[-1, index])
                for index, feature in enumerate(FEATURES)
            },
            intervention,
        )
        current_sequence = np.vstack(
            [current_sequence[1:], _state_to_array(transformed_last)]
        )

        next_state_scaled = _predict_next(forecasting_model, current_sequence, scaler)

        point = {
            "horizon": step,
            "state": {
                feature: float(next_state_scaled[index])
                for index, feature in enumerate(FEATURES)
            },
            "attack_probability": None,
            "attack_category": None,
            "category_confidence": None,
        }
        counterfactual.append(
            _annotate_point(
                point,
                scaled_state=next_state_scaled,
                horizon=step,
                probability_model=probability_model,
                calibrators=calibrators,
                category_estimator=category_estimator,
            )
        )

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
