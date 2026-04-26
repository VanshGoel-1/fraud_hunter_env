"""
Case-bank loader — mounts a per-episode SQLite database and exposes query helpers.

Supports CMS SynPUF schemas:
  - beneficiary_summary
  - inpatient_claims
  - outpatient_claims
  - carrier_claims
  - prescription_drug_events
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CaseHandle:
    case_id: str
    db_path: Path
    conn: sqlite3.Connection
    seen_queries: set[str] = field(default_factory=set)
    tier: int = 1

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ── Ground-truth accessors ────────────────────────────────────────────
    def ground_truth(self, kind: str) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT payload_json FROM ground_truth WHERE kind = ?", (kind,)
        )
        return [json.loads(row[0]) for row in cur.fetchall()]

    def all_entity_names(self) -> set[str]:
        """Every corporate / provider / beneficiary name that exists in the DB."""
        names: set[str] = set()
        for table, col in [
            ("corporate_registry", "entity_name"),
            ("beneficiary_summary", "DESYNPUF_ID"), # Note: SynPUF uses IDs as primary names
        ]:
            try:
                cur = self.conn.execute(f"SELECT {col} FROM {table}")
                names.update(r[0] for r in cur.fetchall() if r[0])
            except sqlite3.OperationalError:
                pass
        return names

    def get_metadata(self, key: str) -> Optional[str]:
        try:
            cur = self.conn.execute(
                "SELECT value FROM case_metadata WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    # ── Agent-facing queries ──────────────────────────────────────────────
    def query_corporate(self, entity_name: Optional[str], entity_id: Optional[str]) -> str:
        if entity_id:
            cur = self.conn.execute(
                "SELECT entity_id, entity_name, tax_id, parent_entity_id, ubo_id, incorporation_date, state, npi_code "
                "FROM corporate_registry WHERE entity_id = ?", (entity_id,),
            )
        else:
            cur = self.conn.execute(
                "SELECT entity_id, entity_name, tax_id, parent_entity_id, ubo_id, incorporation_date, state, npi_code "
                "FROM corporate_registry WHERE entity_name = ? COLLATE NOCASE", (entity_name,),
            )
        rows = cur.fetchall()
        if not rows:
            return f"no_match: corporate_registry has no entity for {entity_name or entity_id!r}"
        
        lines = [
            f"entity_id={r[0]} name={r[1]!r} tax_id={r[2]} parent={r[3]} ubo={r[4]} registered={r[5]} state={r[6]} npi={r[7]}"
            for r in rows
        ]
        return "\n".join(lines)

    def query_medicare(self, beneficiary_id: Optional[str], claim_id: Optional[str]) -> str:
        """Query CMS-aligned health records."""
        if claim_id:
            # Check Carrier Claims first (most common)
            cur = self.conn.execute(
                "SELECT CLM_ID, DESYNPUF_ID, CLM_FROM_DT, PRF_PHYSN_NPI, LINE_NCH_PMT_AMT, HCPCS_CD "
                "FROM carrier_claims WHERE CLM_ID = ?", (claim_id,),
            )
            r = cur.fetchone()
            if r:
                return f"claim_id={r[0]} bene_id={r[1]} date={r[2]} npi={r[3]} amt={r[4]} hcpcs={r[5]}"
            return f"no_match: no claim {claim_id!r} found in carrier_claims"

        # Beneficiary lookup
        bcur = self.conn.execute(
            "SELECT DESYNPUF_ID, BENE_BIRTH_DT, BENE_DEATH_DT, SP_STATE_CODE "
            "FROM beneficiary_summary WHERE DESYNPUF_ID = ?", (beneficiary_id,),
        )
        br = bcur.fetchone()
        if not br:
            return f"no_match: no beneficiary {beneficiary_id!r}"
        
        lines = [f"bene_id={br[0]} dob={br[1]} dod={br[2]} state={br[3]}"]
        
        # List recent claims
        ccur = self.conn.execute(
            "SELECT CLM_ID, CLM_FROM_DT, LINE_NCH_PMT_AMT, HCPCS_CD "
            "FROM carrier_claims WHERE DESYNPUF_ID = ? LIMIT 10", (beneficiary_id,),
        )
        for c in ccur.fetchall():
            lines.append(f"  carrier_claim: id={c[0]} date={c[1]} amt={c[2]} hcpcs={c[3]}")
            
        return "\n".join(lines)


def _demo_case() -> CaseHandle:
    """In-memory demo case following CMS SynPUF schema."""
    # check_same_thread=False — see fraud_hunter_env_environment.py for rationale
    # (the CodeAct sandbox executes user code in a worker thread).
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE beneficiary_summary (
            DESYNPUF_ID TEXT PRIMARY KEY, BENE_BIRTH_DT TEXT, BENE_DEATH_DT TEXT,
            BENE_SEX_IDENT_CD TEXT, BENE_RACE_CD TEXT, SP_STATE_CODE TEXT
        );
        CREATE TABLE carrier_claims (
            CLM_ID TEXT PRIMARY KEY, DESYNPUF_ID TEXT, CLM_FROM_DT TEXT,
            CLM_THRU_DT TEXT, PRF_PHYSN_NPI TEXT, TAX_NUM TEXT,
            HCPCS_CD TEXT, LINE_NCH_PMT_AMT REAL, LINE_ICD9_DGNS_CD TEXT
        );
        CREATE TABLE corporate_registry (
            entity_id TEXT PRIMARY KEY, entity_name TEXT, tax_id TEXT,
            parent_entity_id TEXT, ubo_id TEXT, incorporation_date TEXT,
            state TEXT, npi_code TEXT
        );
        CREATE TABLE ground_truth (kind TEXT, payload_json TEXT);
        CREATE TABLE case_metadata (key TEXT PRIMARY KEY, value TEXT);

        INSERT INTO case_metadata VALUES ('tier', '1');
        INSERT INTO beneficiary_summary VALUES ('BENE_001', '1945-01-01', '2025-11-03', '1', '1', 'CA');
        
        INSERT INTO corporate_registry VALUES 
            ('E001', 'Starpoint Medical LLC', 'TX-001', NULL, 'U_001', '2010-01-01', 'CA', NULL),
            ('E002', 'Starpoint Holdings Ltd', 'TX-002', NULL, 'U_001', '2008-01-01', 'DE', NULL),
            ('E003', 'Acme Shell LLC', 'TX-003', 'E002', 'U_001', '2020-01-01', 'NY', NULL),
            ('E_PROV', 'John Doe MD', 'TX-P', 'E001', 'U_P', '2015-01-01', 'CA', '1234567890');
        
        INSERT INTO carrier_claims VALUES
            ('C001', 'BENE_001', '2026-02-10', '2026-02-10', '1234567890', 'TX-F', '99214', 480.0, 'E11.9'),
            ('C002', 'BENE_001', '2026-02-10', '2026-02-10', '1234567890', 'TX-F', '99214', 480.0, 'E11.9'),
            ('C003', 'BENE_001', '2026-03-01', '2026-03-01', '1234567890', 'TX-F', '99214', 480.0, 'Z00.00');
        
        INSERT INTO ground_truth VALUES
            ('entity', '{"name":"John Doe MD","kind":"provider","npi":"1234567890"}'),
            ('entity', '{"name":"Acme Shell LLC","kind":"corporation"}'),
            ('shell_link', '{"child":"Acme Shell LLC","parent":"Starpoint Holdings Ltd"}'),
            ('contradiction', '{"evidence_a":"claim:C001","evidence_b":"claim:C002","kind":"duplicate_bill"}'),
            ('contradiction', '{"evidence_a":"beneficiary:BENE_001","evidence_b":"claim:C003","kind":"dead_patient_claim"}');
    """)
    conn.commit()
    return CaseHandle(case_id="demo", db_path=Path(":memory:"), conn=conn, tier=1)
