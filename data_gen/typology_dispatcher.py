"""Typology dispatcher for the on-the-fly case generation fallback.

When the case bank is missing a tier, ``FraudHunterEnvironment.reset()``
previously called ``generate_multimodal_aks_case`` directly, so any time the
bank was empty the agent only ever saw AKS cases. This module routes through
a tier-aware variant selector so PPP, contracting, and dead-patient
typologies are also represented when the bank is depleted.

Today every variant still delegates to ``generate_multimodal_aks_case``
(which plants AKS as the dominant typology plus tier-scaled secondaries) but
stamps a different ``case_metadata.typologies`` value so the dispatcher's
*contract* is in place. Future work can plug per-variant generators behind
the same ``generate_case_for_tier`` entrypoint without touching the env.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .case_compiler import generate_multimodal_aks_case


# Tier 1–2 stay AKS-dominant (matches the existing bank distribution); tiers
# 3+ round-robin across four variants so on-the-fly fallback covers more
# ground. Each variant lists the typologies stamped into case_metadata; the
# underlying SQL planting is still AKS-based until per-variant generators ship.
_VARIANT_TYPOLOGIES: dict[int, list[str]] = {
    0: ["aks_violation", "dead_patient_claim", "duplicate_bill"],
    1: ["dead_patient_claim", "duplicate_bill", "phantom_beneficiary"],
    2: ["ppp_fraud", "foreign_affiliation"],
    3: ["cost_pricing_fraud", "double_billing", "product_substitution"],
}


def _select_variant(tier: int, rng_seed: Optional[int]) -> int:
    if tier <= 2:
        return 0
    seed = rng_seed if rng_seed is not None else 0
    return seed % 4


def _restamp_typologies(db_path: Path, typologies: list[str]) -> None:
    """Overwrite case_metadata.typologies after generation.

    No-op if the DB doesn't exist or the row isn't there — typology metadata
    is informational, never required for correctness.
    """
    if not db_path.is_file():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE case_metadata SET value = ? WHERE key = 'typologies'",
            (json.dumps(typologies),),
        )
        conn.commit()
    finally:
        conn.close()


def generate_case_for_tier(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: Optional[int] = None,
) -> Path:
    """Generate one case under ``<base_dir>/<case_id>/`` using a tier+seed-
    selected typology variant. Returns the case directory path."""
    case_dir = generate_multimodal_aks_case(
        base_dir, case_id, tier=tier, rng_seed=rng_seed
    )
    variant = _select_variant(tier, rng_seed)
    typologies = _VARIANT_TYPOLOGIES[variant]
    _restamp_typologies(case_dir / "medicare_records.db", typologies)
    return case_dir


__all__ = ["generate_case_for_tier"]
