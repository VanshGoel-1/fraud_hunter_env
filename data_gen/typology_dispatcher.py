"""Typology dispatcher for the on-the-fly case generation fallback.

When the case bank is missing a tier, ``FraudHunterEnvironment.reset()``
calls ``generate_case_for_tier`` which routes through a tier-aware variant
selector so PPP, shell-chain, and dead-patient typologies are also represented
in addition to the AKS multimodal cases.

Tier 1–2 always use variant 0 (AKS-dominant, matching the bank distribution).
Tier 3+ round-robins across four variants keyed by (seed % 4):
  0 — AKS multimodal (default)
  1 — Dead-patient claim only
  2 — PPP loan fraud only
  3 — Shell chain only (link_shell signals, no billing anomalies)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .case_compiler import (
    generate_dead_patient_case,
    generate_multimodal_aks_case,
    generate_ppp_fraud_case,
    generate_shell_chain_case,
)


def _select_variant(tier: int, rng_seed: Optional[int]) -> int:
    if tier <= 2:
        return 0
    seed = rng_seed if rng_seed is not None else 0
    return seed % 4


def generate_case_for_tier(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: Optional[int] = None,
) -> Path:
    """Generate one case under ``<base_dir>/<case_id>/`` using a tier+seed-
    selected typology variant. Returns the case directory path."""
    variant = _select_variant(tier, rng_seed)
    if variant == 1:
        return generate_dead_patient_case(base_dir, case_id, tier=tier, rng_seed=rng_seed)
    if variant == 2:
        return generate_ppp_fraud_case(base_dir, case_id, tier=tier, rng_seed=rng_seed)
    if variant == 3:
        return generate_shell_chain_case(base_dir, case_id, tier=tier, rng_seed=rng_seed)
    return generate_multimodal_aks_case(base_dir, case_id, tier=tier, rng_seed=rng_seed)


__all__ = ["generate_case_for_tier"]
