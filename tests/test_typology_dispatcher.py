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


def _has_table_rows(case_dir: Path, table: str) -> bool:
    conn = sqlite3.connect(str(case_dir / "medicare_records.db"))
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False
    finally:
        conn.close()


def _has_death_date_claim(case_dir: Path) -> bool:
    """True if any carrier_claim was filed after the beneficiary's BENE_DEATH_DT."""
    conn = sqlite3.connect(str(case_dir / "medicare_records.db"))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM carrier_claims cc "
            "JOIN beneficiary_summary bs ON cc.DESYNPUF_ID = bs.DESYNPUF_ID "
            "WHERE bs.BENE_DEATH_DT IS NOT NULL "
            "AND cc.CLM_FROM_DT > bs.BENE_DEATH_DT"
        ).fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False
    finally:
        conn.close()


def test_dispatcher_diversifies_high_tier():
    """Seeds 300-303 at tier 3 map to variants 0-3 (300%4=0, 301%4=1, etc.).
    Each variant produces genuinely different DB content:
      variant 0 (seed 300): AKS → has prescription_drug_events fraud rows
      variant 1 (seed 301): dead-patient → carrier_claim after BENE_DEATH_DT
      variant 2 (seed 302): PPP fraud → has loan_applications rows
      variant 3 (seed 303): shell-chain → corporate_registry with parent links, no loan_applications
    """
    with tempfile.TemporaryDirectory() as td:
        # variant 1 — dead-patient: a carrier claim filed after death date
        dead_dir = generate_case_for_tier(Path(td), "case_dead", tier=3, rng_seed=301)
        assert _has_death_date_claim(dead_dir), "variant 1 must have a post-death carrier claim"

        # variant 2 — PPP fraud: must have loan_applications rows
        ppp_dir = generate_case_for_tier(Path(td), "case_ppp", tier=3, rng_seed=302)
        assert _has_table_rows(ppp_dir, "loan_applications"), \
            "variant 2 must have loan_applications rows"

        # variant 3 — shell-chain: must have corporate_registry parent links
        shell_dir = generate_case_for_tier(Path(td), "case_shell", tier=3, rng_seed=303)
        conn = sqlite3.connect(str(shell_dir / "medicare_records.db"))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM corporate_registry WHERE parent_entity_id IS NOT NULL"
            ).fetchone()
            assert row and row[0] >= 3, "variant 3 must have ≥3 shell entities with parents"
        finally:
            conn.close()


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
