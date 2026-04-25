"""
Case-bank contract tests.

These tests pin down the *structural* contract that every generated case
must satisfy at each tier — they fail fast if the case compiler is changed
in a way that breaks the agent's assumptions about what evidence is present.

For every (tier, seed) combination we instantiate the compiler directly into
a tmp dir, open the SQLite DB, and assert structural invariants:

  1. typology curve floor — each tier produces at least the minimum number
     of distinct typologies promised in the difficulty spec.
  2. corporate-registry size — each tier seeds at least N entities.
  3. shell-chain presence at tier ≥ 3 — non-null parent_entity_id rows.
  4. government-contracts present at tier ≥ 4.
  5. PPP loans + foreign affiliations present at tier 5.

Each contract runs across multiple seeds so we catch RNG-dependent regressions.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fraud_hunter_env.data_gen.case_compiler import generate_multimodal_aks_case


SEEDS = (17, 42, 123, 1009)


# Minimum typology counts per tier (floor — generator may add more).
_TIER_TYPOLOGY_FLOOR = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Minimum corporate_registry row counts per tier.
_TIER_REGISTRY_FLOOR = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}


def _build_case(tmp_path: Path, tier: int, seed: int) -> sqlite3.Connection:
    case_id = f"contract_t{tier}_s{seed}"
    generate_multimodal_aks_case(tmp_path, case_id, tier=tier, rng_seed=seed)
    db_path = tmp_path / case_id / "medicare_records.db"
    assert db_path.is_file(), f"compiler did not produce a DB at {db_path}"
    return sqlite3.connect(str(db_path))


def _typology_set(conn: sqlite3.Connection) -> set[str]:
    """Read typologies from case_metadata['typologies'] JSON list."""
    cur = conn.execute(
        "SELECT value FROM case_metadata WHERE key = 'typologies'"
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return set()
    try:
        return set(json.loads(row[0]))
    except (TypeError, ValueError):
        return set()


# ─── Contract 1: typology floor per tier ────────────────────────────────────

@pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_typology_floor_per_tier(tmp_path: Path, tier: int, seed: int) -> None:
    conn = _build_case(tmp_path, tier, seed)
    try:
        typs = _typology_set(conn)
        floor = _TIER_TYPOLOGY_FLOOR[tier]
        assert len(typs) >= floor, (
            f"tier={tier} seed={seed}: expected ≥{floor} typologies, "
            f"got {len(typs)}: {sorted(typs)}"
        )
    finally:
        conn.close()


# ─── Contract 2: corporate registry size ────────────────────────────────────

@pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_corporate_registry_size(tmp_path: Path, tier: int, seed: int) -> None:
    conn = _build_case(tmp_path, tier, seed)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM corporate_registry"
        ).fetchone()
        floor = _TIER_REGISTRY_FLOOR[tier]
        assert row[0] >= floor, (
            f"tier={tier} seed={seed}: corporate_registry has {row[0]} rows, "
            f"expected ≥{floor}"
        )
    finally:
        conn.close()


# ─── Contract 3: shell chain (parent_entity_id) at tier ≥ 3 ─────────────────

@pytest.mark.parametrize("tier", [3, 4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_shell_chain_present(tmp_path: Path, tier: int, seed: int) -> None:
    conn = _build_case(tmp_path, tier, seed)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM corporate_registry "
            "WHERE parent_entity_id IS NOT NULL AND parent_entity_id != ''"
        ).fetchone()
        assert row[0] >= 1, (
            f"tier={tier} seed={seed}: expected at least one shell link "
            f"(parent_entity_id != NULL), got {row[0]}"
        )
    finally:
        conn.close()


# ─── Contract 4: government contracts at tier ≥ 4 ───────────────────────────

@pytest.mark.parametrize("tier", [4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_government_contracts_present(tmp_path: Path, tier: int, seed: int) -> None:
    conn = _build_case(tmp_path, tier, seed)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM government_contracts"
        ).fetchone()
        assert row[0] >= 1, (
            f"tier={tier} seed={seed}: expected ≥1 government contract row, "
            f"got {row[0]}"
        )
    finally:
        conn.close()


# ─── Contract 5: PPP + foreign at tier 5 ────────────────────────────────────

@pytest.mark.parametrize("seed", SEEDS)
def test_tier5_ppp_and_foreign(tmp_path: Path, seed: int) -> None:
    conn = _build_case(tmp_path, 5, seed)
    try:
        loans = conn.execute(
            "SELECT COUNT(*) FROM loan_applications"
        ).fetchone()[0]
        foreign = conn.execute(
            "SELECT COUNT(*) FROM foreign_affiliations"
        ).fetchone()[0]
        assert loans >= 1, f"tier=5 seed={seed}: expected ≥1 PPP loan, got {loans}"
        assert foreign >= 1, (
            f"tier=5 seed={seed}: expected ≥1 foreign_affiliations row, got {foreign}"
        )
    finally:
        conn.close()
