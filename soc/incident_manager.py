from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.database import SessionLocal
from backend.models.incident import (
    IncidentLifecycleModel,
    IncidentModel,
)

from soc.incident import (
    Incident,
    InvalidLifecycleTransitionError,
    LIFECYCLE_STAGES,
    STATUS_ORDER,
)


class DuplicateIncidentError(ValueError):
    """Raised when an incident with the same ID already exists."""


class IncidentNotFoundError(KeyError):
    """Raised when an incident cannot be found."""


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _normalize_json_value(value: Any) -> Any:
    """
    Convert JSON-serialized strings back into Python JSON-compatible
    objects before storing them in PostgreSQL JSONB columns.

    Some existing ThreatMind SOC objects return JSON fields as strings,
    while PostgreSQL JSONB expects Python dictionaries/lists.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def _model_to_dict(
    incident: IncidentModel,
    lifecycle: Optional[List[IncidentLifecycleModel]] = None,
) -> Dict[str, Any]:
    """
    Convert a PostgreSQL IncidentModel into the dictionary shape
    expected by the existing API and tests.
    """

    if lifecycle is None:
        lifecycle = []

    lifecycle_history = [
        {
            "status": item.status,
            "timestamp": item.timestamp.isoformat(),
        }
        for item in sorted(
            lifecycle,
            key=lambda item: item.timestamp,
        )
    ]

    return {
        "incident_id": incident.incident_id,
        "timestamp": incident.timestamp,
        "status": incident.status,
        "created_at": incident.created_at.isoformat(),
        "updated_at": incident.updated_at.isoformat(),
        "risk_level": incident.risk_level,
        "peak_attack_probability": incident.peak_attack_probability,
        "peak_horizon": incident.peak_horizon,
        "attack_category": incident.attack_category,
        "recommended_intervention": incident.recommended_intervention,
        "decision_basis": incident.decision_basis,
        "mitre_techniques": incident.mitre_techniques,
        "uncertainties": incident.uncertainties,
        "evidence": incident.evidence,
        "lifecycle_history": lifecycle_history,
    }


def clear_incidents() -> None:
    """
    Delete all incidents and their lifecycle records.

    Primarily used by tests.
    """

    db = SessionLocal()

    try:
        db.query(IncidentLifecycleModel).delete(
            synchronize_session=False
        )

        db.query(IncidentModel).delete(
            synchronize_session=False
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def create_incident(
    soc_analysis: Dict[str, Any],
    initial_status: str = "RECOMMENDED",
) -> Dict[str, Any]:
    """
    Create and persist a new incident from SOC analysis.

    The default remains RECOMMENDED for backward compatibility.

    The API explicitly passes DETECTED when it wants the incident
    lifecycle to begin at the detection stage.
    """

    incident = Incident.from_soc_analysis(
        soc_analysis,
        initial_status=initial_status,
    )

    data = incident.to_dict()

    incident_id = data["incident_id"]

    db = SessionLocal()

    try:
        existing = (
            db.query(IncidentModel)
            .filter(
                IncidentModel.incident_id == incident_id
            )
            .first()
        )

        if existing is not None:
            raise DuplicateIncidentError(
                f"Incident '{incident_id}' already exists."
            )

        created_at = _utc_now()

        db_incident = IncidentModel(
            incident_id=data["incident_id"],
            timestamp=data["timestamp"],
            status=data["status"],
            created_at=created_at,
            updated_at=created_at,
            risk_level=data["risk_level"],
            peak_attack_probability=data[
                "peak_attack_probability"
            ],
            peak_horizon=data["peak_horizon"],
            attack_category=data["attack_category"],

            # JSONB fields
            recommended_intervention=_normalize_json_value(
                data["recommended_intervention"]
            ),

            decision_basis=_normalize_json_value(
                data["decision_basis"]
            ),

            mitre_techniques=_normalize_json_value(
                data["mitre_techniques"]
            ),

            uncertainties=_normalize_json_value(
                data["uncertainties"]
            ),

            evidence=_normalize_json_value(
                data["evidence"]
            ),
        )

        db.add(db_incident)

        lifecycle_timestamp = created_at

        db_lifecycle = IncidentLifecycleModel(
            incident_id=incident_id,
            status=data["status"],
            timestamp=lifecycle_timestamp,
        )

        db.add(db_lifecycle)

        db.commit()

        db.refresh(db_incident)

        lifecycle = (
            db.query(IncidentLifecycleModel)
            .filter(
                IncidentLifecycleModel.incident_id
                == incident_id
            )
            .order_by(
                IncidentLifecycleModel.timestamp.asc(),
                IncidentLifecycleModel.id.asc(),
            )
            .all()
        )

        return _model_to_dict(
            db_incident,
            lifecycle,
        )

    except DuplicateIncidentError:
        db.rollback()
        raise

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_incident(
    incident_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve one incident by its public incident ID.
    """

    db = SessionLocal()

    try:
        incident = (
            db.query(IncidentModel)
            .filter(
                IncidentModel.incident_id == incident_id
            )
            .first()
        )

        if incident is None:
            return None

        lifecycle = (
            db.query(IncidentLifecycleModel)
            .filter(
                IncidentLifecycleModel.incident_id
                == incident_id
            )
            .order_by(
                IncidentLifecycleModel.timestamp.asc(),
                IncidentLifecycleModel.id.asc(),
            )
            .all()
        )

        return _model_to_dict(
            incident,
            lifecycle,
        )

    finally:
        db.close()


def list_incidents() -> List[Dict[str, Any]]:
    """
    Return all incidents ordered newest first.
    """

    db = SessionLocal()

    try:
        incidents = (
            db.query(IncidentModel)
            .order_by(
                IncidentModel.created_at.desc(),
                IncidentModel.id.desc(),
            )
            .all()
        )

        if not incidents:
            return []

        incident_ids = [
            incident.incident_id
            for incident in incidents
        ]

        lifecycle_rows = (
            db.query(IncidentLifecycleModel)
            .filter(
                IncidentLifecycleModel.incident_id.in_(
                    incident_ids
                )
            )
            .order_by(
                IncidentLifecycleModel.timestamp.asc(),
                IncidentLifecycleModel.id.asc(),
            )
            .all()
        )

        lifecycle_by_incident: Dict[
            str,
            List[IncidentLifecycleModel],
        ] = {}

        for row in lifecycle_rows:
            lifecycle_by_incident.setdefault(
                row.incident_id,
                [],
            ).append(row)

        return [
            _model_to_dict(
                incident,
                lifecycle_by_incident.get(
                    incident.incident_id,
                    [],
                ),
            )
            for incident in incidents
        ]

    finally:
        db.close()


def update_incident_status(
    incident_id: str,
    target_status: str,
) -> Dict[str, Any]:
    """
    Advance an incident exactly one lifecycle stage.

    Valid lifecycle:

        DETECTED
            ↓
        ASSESSED
            ↓
        FORECASTED
            ↓
        SIMULATED
            ↓
        RECOMMENDED

    Backward transitions and skipped stages are rejected.
    RECOMMENDED is terminal.
    """

    target_status = target_status.upper().strip()

    if target_status not in LIFECYCLE_STAGES:
        raise InvalidLifecycleTransitionError(
            f"Unknown lifecycle status: '{target_status}'."
        )

    db = SessionLocal()

    try:
        incident = (
            db.query(IncidentModel)
            .filter(
                IncidentModel.incident_id == incident_id
            )
            .first()
        )

        if incident is None:
            raise IncidentNotFoundError(
                f"Incident '{incident_id}' not found."
            )

        current_status = incident.status

        if current_status == "RECOMMENDED":
            raise InvalidLifecycleTransitionError(
                "RECOMMENDED is a terminal incident state."
            )

        current_index = STATUS_ORDER[current_status]
        target_index = STATUS_ORDER[target_status]

        expected_index = current_index + 1

        if target_index != expected_index:
            if target_index <= current_index:
                raise InvalidLifecycleTransitionError(
                    f"Cannot move incident from "
                    f"{current_status} back to "
                    f"{target_status}."
                )

            raise InvalidLifecycleTransitionError(
                f"Cannot skip lifecycle stage: "
                f"{current_status} -> "
                f"{target_status}. "
                f"Expected "
                f"{LIFECYCLE_STAGES[expected_index]}."
            )

        now = _utc_now()

        incident.status = target_status
        incident.updated_at = now

        lifecycle = IncidentLifecycleModel(
            incident_id=incident_id,
            status=target_status,
            timestamp=now,
        )

        db.add(lifecycle)

        db.commit()

        db.refresh(incident)

        lifecycle_rows = (
            db.query(IncidentLifecycleModel)
            .filter(
                IncidentLifecycleModel.incident_id
                == incident_id
            )
            .order_by(
                IncidentLifecycleModel.timestamp.asc(),
                IncidentLifecycleModel.id.asc(),
            )
            .all()
        )

        return _model_to_dict(
            incident,
            lifecycle_rows,
        )

    except (
        IncidentNotFoundError,
        InvalidLifecycleTransitionError,
    ):
        db.rollback()
        raise

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


__all__ = [
    "DuplicateIncidentError",
    "IncidentNotFoundError",
    "clear_incidents",
    "create_incident",
    "get_incident",
    "list_incidents",
    "update_incident_status",
]