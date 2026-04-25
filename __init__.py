"""
Government Fraud Hunter AI — OpenEnv-compliant RL environment
for qui tam fraud investigations.

Exports the client, action/observation models, and key enums.
"""

from .models import (
    ActionKind,
    ContradictionKind,
    EntityKind,
    FraudHunterAction,
    FraudHunterObservation,
)
from .client import FraudHunterEnv

__all__ = [
    "ActionKind",
    "ContradictionKind",
    "EntityKind",
    "FraudHunterAction",
    "FraudHunterObservation",
    "FraudHunterEnv",
]
