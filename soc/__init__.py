"""ThreatMind SOC Package.

Provides end-to-end orchestration, structured response formatting,
evidence-tiered explainability, and incident lifecycle management.

Heavy submodules (soc_orchestrator, soc_response, soc_explanation) are NOT
imported at package level because they transitively depend on torch/sqlalchemy.
Import them directly when needed:

    from soc.soc_orchestrator import run_soc_analysis
    from soc.soc_response import build_soc_response
    from soc.soc_explanation import generate_soc_explanation
"""

from soc.incident import (
    LIFECYCLE_STAGES,
    VALID_TRANSITIONS,
    Incident,
    InvalidLifecycleTransitionError,
    generate_incident_id,
)
from soc.incident_manager import (
    DuplicateIncidentError,
    IncidentNotFoundError,
    clear_incidents,
    create_incident,
    get_incident,
    list_incidents,
    update_incident_status,
)

__all__ = [
    "LIFECYCLE_STAGES",
    "VALID_TRANSITIONS",
    "DuplicateIncidentError",
    "Incident",
    "IncidentNotFoundError",
    "InvalidLifecycleTransitionError",
    "clear_incidents",
    "create_incident",
    "generate_incident_id",
    "get_incident",
    "list_incidents",
    "update_incident_status",
]
