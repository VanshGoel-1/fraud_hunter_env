"""Tests for the on-the-fly typology dispatcher.

Confirms that the dispatcher produces typology-varying cases when the case
bank fallback is exercised — the bug it replaces collapsed every fallback
case to AKS regardless of tier or seed.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

from fraud_hunter_env.data_gen.typology_dispatcher import generate_case_for_tier


def _read_typologies(case_dir: Path) -> list[str]:
    conn = sqlite3.connect(str(case_dir / "medicare_records.db"))
    try:
        row = conn.execute(
            "SELECT value FROM case_metadata WHERE key='typologies'"
        ).fetchone()
        assert row is not None and row[0], "case_metadata.typologies missing"
        return json.loads(row[0])
    finally:
        conn.close()


def test_dispatcher_low_tier_keeps_aks_dominant():
    with tempfile.TemporaryDirectory() as td:
        case_dir = generate_case_for_tier(
            Path(td), "case_low", tier=1, rng_seed=42,
        )
        typs = _read_typologies(case_dir)
        assert "aks_violation" in typs


def test_dispatcher_diversifies_high_tier():
    """Across 4 distinct seeds at tier 3+, we should see >1 distinct dominant
    typology. Each variant's typologies list is unique by design, so any two
    different (seed % 4) values guarantee distinct typology coverage."""
    seen: set[str] = set()
    with tempfile.TemporaryDirectory() as td:
        for i, seed in enumerate([300, 301, 302, 303]):
            case_dir = generate_case_for_tier(
                Path(td), f"case_{i}", tier=3, rng_seed=seed,
            )
            seen.update(_read_typologies(case_dir))
    assert len(seen) >= 2, f"only {len(seen)} typologies seen across variants: {seen}"


def test_dispatcher_returns_case_dir():
    with tempfile.TemporaryDirectory() as td:
        case_dir = generate_case_for_tier(
            Path(td), "case_x", tier=2, rng_seed=7,
        )
        assert case_dir.is_dir()
        assert (case_dir / "medicare_records.db").is_file()


def test_dispatcher_tier_2_still_aks_dominant():
    """Tier 1–2 should always select variant 0 regardless of seed."""
    with tempfile.TemporaryDirectory() as td:
        for seed in [1, 2, 3, 99]:
            case_dir = generate_case_for_tier(
                Path(td), f"case_{seed}", tier=2, rng_seed=seed,
            )
            typs = _read_typologies(case_dir)
            assert "aks_violation" in typs
