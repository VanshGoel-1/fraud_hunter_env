"""
Case Compiler — Multi-modal synthetic fraud cases (CMS SynPUF + messy evidence).

Per-case output is a directory:

    <base_dir>/<case_id>/
        medicare_records.db          # SQLite, CMS SynPUF-aligned
        intercepted_comms/           # .txt emails (50 benign + 1 smoking gun)
        scanned_claims/              # degraded CMS-1500 single-image PDFs

Primary typology: Anti-Kickback Statute (AKS) kickback disguised as a
"research grant" — the smoking gun lives in intercepted_comms/ (needle in
haystack, OCR-noised NPI), and the crime lives in the PDE + carrier_claims
tables (anomalous volume spike + upcoded claim with a degraded PDF).

Secondary typologies: shell-company chain (tier-scaled depth), dead-patient
claim, duplicate billing. Ground-truth rows are planted into the
`ground_truth(kind, payload_json)` table the grader already reads from.

Quality primitives (kept from the previous CMS-only compiler):
    - Luhn-valid NPIs with 80840 prefix
    - Benford's-law-shaped dollar amounts
    - Pareto heavy-tail amounts (optional)
    - CMS DE-SynPUF field names (DESYNPUF_ID, BENE_BIRTH_DT, etc.)
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fraud_hunter_env.npi_utils import generate_valid_npi

from .pdf_evidence import ClaimEvidence, render_claim_pdf


# ── Benford & Pareto sampling ────────────────────────────────────────────────

_BENFORD = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]


def _benford_amount(rng: random.Random, lo: float = 50.0, hi: float = 5000.0) -> float:
    first = rng.choices(range(1, 10), weights=_BENFORD, k=1)[0]
    for _ in range(1000):
        exp = rng.uniform(math.log10(max(1.0, lo)), math.log10(max(2.0, hi)))
        raw = 10 ** exp
        s = str(round(raw, 2)).lstrip("0.")
        if s and int(s[0]) == first and lo <= raw <= hi:
            return round(raw, 2)
    return round(rng.uniform(lo, hi), 2)


def _pareto_amount(rng: random.Random, xm: float = 1000.0, alpha: float = 1.5) -> float:
    u = rng.random()
    return round(xm / (u ** (1 / alpha)), 2)


def _rand_date(rng: random.Random, start: str, end: str) -> str:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    return (s + timedelta(days=rng.randint(0, max(delta, 1)))).strftime("%Y-%m-%d")


# ── OCR noise for unstructured evidence ──────────────────────────────────────

_OCR_NOISE_MAP = {"0": "O", "1": "l", "5": "S", "8": "B"}


def _inject_ocr_noise(text: str, rng: random.Random, n_errors: int = 2) -> str:
    """Replace digits with visually-similar glyphs to force agent data-cleaning."""
    chars = list(text)
    candidates = [i for i, c in enumerate(chars) if c in _OCR_NOISE_MAP]
    if not candidates:
        return text
    rng.shuffle(candidates)
    for idx in candidates[:max(1, n_errors)]:
        chars[idx] = _OCR_NOISE_MAP[chars[idx]]
    return "".join(chars)


# ── CMS SynPUF schema ────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE beneficiary_summary (
    DESYNPUF_ID TEXT PRIMARY KEY,
    BENE_BIRTH_DT TEXT,
    BENE_DEATH_DT TEXT,
    BENE_SEX_IDENT_CD TEXT,
    BENE_RACE_CD TEXT,
    BENE_ESRD_IND TEXT,
    SP_STATE_CODE TEXT,
    BENE_COUNTY_CD TEXT,
    SP_ALZHDMTA INTEGER DEFAULT 0,
    SP_CHF INTEGER DEFAULT 0,
    SP_CHRNKIDN INTEGER DEFAULT 0,
    SP_CNCR INTEGER DEFAULT 0,
    SP_COPD INTEGER DEFAULT 0,
    SP_DEPRESSN INTEGER DEFAULT 0,
    SP_DIABETES INTEGER DEFAULT 0,
    SP_ISCHMCHT INTEGER DEFAULT 0,
    SP_OSTEOPRS INTEGER DEFAULT 0,
    SP_RA_OA INTEGER DEFAULT 0,
    SP_STRKETIA INTEGER DEFAULT 0,
    MEDREIMB_IP REAL DEFAULT 0,
    BENRES_IP REAL DEFAULT 0,
    PPPYMT_IP REAL DEFAULT 0
);

CREATE TABLE inpatient_claims (
    CLM_ID TEXT PRIMARY KEY,
    DESYNPUF_ID TEXT,
    CLM_FROM_DT TEXT,
    CLM_THRU_DT TEXT,
    PRVDR_NUM TEXT,
    CLM_PMT_AMT REAL,
    AT_PHYSN_NPI TEXT,
    OP_PHYSN_NPI TEXT,
    OT_PHYSN_NPI TEXT,
    CLM_ADMSN_DT TEXT,
    ADMTNG_ICD9_DGNS_CD TEXT,
    CLM_DRG_CD TEXT,
    ICD9_DGNS_CD_1 TEXT,
    ICD9_PRCDR_CD_1 TEXT,
    FOREIGN KEY(DESYNPUF_ID) REFERENCES beneficiary_summary(DESYNPUF_ID)
);

CREATE TABLE outpatient_claims (
    CLM_ID TEXT PRIMARY KEY,
    DESYNPUF_ID TEXT,
    CLM_FROM_DT TEXT,
    CLM_THRU_DT TEXT,
    PRVDR_NUM TEXT,
    CLM_PMT_AMT REAL,
    AT_PHYSN_NPI TEXT,
    OP_PHYSN_NPI TEXT,
    OT_PHYSN_NPI TEXT,
    ADMTNG_ICD9_DGNS_CD TEXT,
    ICD9_DGNS_CD_1 TEXT,
    HCPCS_CD_1 TEXT,
    FOREIGN KEY(DESYNPUF_ID) REFERENCES beneficiary_summary(DESYNPUF_ID)
);

CREATE TABLE carrier_claims (
    CLM_ID TEXT PRIMARY KEY,
    DESYNPUF_ID TEXT,
    CLM_FROM_DT TEXT,
    CLM_THRU_DT TEXT,
    PRF_PHYSN_NPI TEXT,
    TAX_NUM TEXT,
    HCPCS_CD TEXT,
    LINE_NCH_PMT_AMT REAL,
    LINE_ICD9_DGNS_CD TEXT,
    FOREIGN KEY(DESYNPUF_ID) REFERENCES beneficiary_summary(DESYNPUF_ID)
);

CREATE TABLE prescription_drug_events (
    PDE_ID TEXT PRIMARY KEY,
    DESYNPUF_ID TEXT,
    PRVDR_NPI TEXT,
    SRVC_DT TEXT,
    PROD_SRVC_ID TEXT,
    DRUG_NAME TEXT,
    QTY_DSPNSD_NUM REAL,
    DAYS_SUPLY_NUM INTEGER,
    PTNT_PAY_AMT REAL,
    TOT_RX_CST_AMT REAL,
    FOREIGN KEY(DESYNPUF_ID) REFERENCES beneficiary_summary(DESYNPUF_ID)
);

CREATE TABLE corporate_registry (
    entity_id TEXT PRIMARY KEY,
    entity_name TEXT,
    tax_id TEXT,
    parent_entity_id TEXT,
    ubo_id TEXT,
    incorporation_date TEXT,
    state TEXT,
    npi_code TEXT,
    FOREIGN KEY(parent_entity_id) REFERENCES corporate_registry(entity_id)
);

CREATE TABLE general_ledger (
    tx_id TEXT PRIMARY KEY,
    tx_date TEXT,
    debit_account TEXT,
    credit_account TEXT,
    amount REAL,
    memo TEXT,
    entity_id TEXT,
    FOREIGN KEY(entity_id) REFERENCES corporate_registry(entity_id)
);

CREATE TABLE referral_payments (
    payment_id TEXT PRIMARY KEY,
    payer_npi TEXT,
    payee_npi TEXT,
    amount REAL,
    payment_date TEXT,
    memo TEXT
);

CREATE TABLE ground_truth (kind TEXT, payload_json TEXT);
CREATE TABLE case_metadata (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE evidence_documents (
    doc_id              TEXT PRIMARY KEY,
    claim_id            TEXT,
    pdf_path            TEXT,
    tier                INTEGER,
    is_scanned          INTEGER DEFAULT 1,
    expected_fields_json TEXT,
    FOREIGN KEY(claim_id) REFERENCES carrier_claims(CLM_ID)
);

-- ── Government contracting domain ───────────────────────────────────────
CREATE TABLE government_contracts (
    contract_id         TEXT PRIMARY KEY,
    agency              TEXT,
    vendor_entity_id    TEXT,
    contract_value      REAL,
    award_date          TEXT,
    expected_product    TEXT,
    disclosed_unit_price REAL,
    actual_unit_cost    REAL,
    FOREIGN KEY(vendor_entity_id) REFERENCES corporate_registry(entity_id)
);

CREATE TABLE contract_invoices (
    invoice_id          TEXT PRIMARY KEY,
    contract_id         TEXT,
    line_item           TEXT,
    amount              REAL,
    invoice_date        TEXT,
    FOREIGN KEY(contract_id) REFERENCES government_contracts(contract_id)
);

CREATE TABLE contract_deliveries (
    delivery_id         TEXT PRIMARY KEY,
    contract_id         TEXT,
    delivered_product   TEXT,
    delivery_date       TEXT,
    FOREIGN KEY(contract_id) REFERENCES government_contracts(contract_id)
);

-- ── PPP / pandemic loan domain ──────────────────────────────────────────
CREATE TABLE loan_applications (
    loan_id             TEXT PRIMARY KEY,
    entity_id           TEXT,
    program             TEXT,
    claimed_employees   INTEGER,
    claimed_monthly_payroll REAL,
    loan_amount         REAL,
    application_date    TEXT,
    FOREIGN KEY(entity_id) REFERENCES corporate_registry(entity_id)
);

CREATE TABLE payroll_records (
    payroll_id          TEXT PRIMARY KEY,
    entity_id           TEXT,
    period_end          TEXT,
    employee_count      INTEGER,
    total_payroll       REAL,
    FOREIGN KEY(entity_id) REFERENCES corporate_registry(entity_id)
);

-- ── Foreign-affiliation disclosure ──────────────────────────────────────
CREATE TABLE foreign_affiliations (
    entity_id           TEXT PRIMARY KEY,
    foreign_parent_name TEXT,
    foreign_country     TEXT,
    disclosure_status   TEXT,
    FOREIGN KEY(entity_id) REFERENCES corporate_registry(entity_id)
);
"""

_ICD9_CODES = ["401.1", "428.0", "250.0", "414.0", "491.2", "311", "715.0"]
_HCPCS_CODES = ["99213", "99214", "99215", "99203", "99204"]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
_LEGIT_DRUGS = ["Lisinopril", "Atorvastatin", "Metformin", "Amlodipine", "Levothyroxine"]
_SCRUTINIZED_DRUGS = ["OxyContin", "Subsys", "Suboxone", "Fentanyl", "Adderall"]
_PHARMA_SHELLS = ["Apex Research Partners", "Helix Consulting Group", "Meridian Scientific LLC",
                  "Solara Advisors", "Beacon Clinical Partners"]
_FRAUD_DOCTORS = ["Dr. Aris Thorne", "Dr. Marcus Vale", "Dr. Celia Drake", "Dr. Julian Crane",
                  "Dr. Helen Voss"]

# Gov contracting / PPP / foreign affiliation references
_AGENCIES = ["DoD", "VA", "GSA", "HHS", "DHS", "DoE"]
_FRAUD_CONTRACTORS = ["Vanguard Logistics LLC", "Sentinel Defense Group",
                      "Atlas Procurement Corp", "Polaris Industrial Inc",
                      "Crestline Supply Co"]
_PRODUCTS_EXPECTED = ["Kevlar Vest Plate", "Field Radio v3", "Ruggedised Laptop",
                      "Surgical Tray Kit", "Backup Generator 5kW"]
_PRODUCTS_SUBSTITUTED = ["Generic Vest Plate", "Discontinued Radio v1",
                         "Refurbished Laptop", "Open Surgical Tray",
                         "Used Generator 3kW"]
_FOREIGN_PARENTS = [("Tianjin Holdings Group", "CN"),
                    ("Volga Industrial OAO", "RU"),
                    ("Caspian Trading FZE", "AE"),
                    ("Pyongyang Heavy Industries", "KP"),
                    ("Caracas Enterprises SA", "VE")]
# CPT bundles where unbundling is the textbook fraud pattern.
# The "bundle_code" should be billed instead of the components.
_UNBUNDLING_PAIRS = [
    ("80053", ["80048", "84520", "84443"]),  # comprehensive metabolic panel
    ("80061", ["83718", "83721", "84478"]),  # lipid panel
    ("99213", ["99202", "36415"]),           # office visit + draw
]

# Off-label marketing: drug → its FDA-approved indication ICD code(s).
# Billing the drug for any other ICD is the off-label red flag.
_OFFLABEL_DRUGS = {
    "Subsys":   {"approved_icd": ["199.0", "199.1", "162.9"]},  # cancer pain
    "Suboxone": {"approved_icd": ["304.00", "304.10"]},         # opioid use disorder
    "OxyContin":{"approved_icd": ["199.0", "338.3"]},           # cancer / chronic pain
}


# ── Public entry point ───────────────────────────────────────────────────────

def generate_multimodal_aks_case(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: int | None = None,
) -> Path:
    """
    Build one multi-modal fraud case under ``<base_dir>/<case_id>/``.

    Returns the case directory path.
    """
    base_dir = Path(base_dir)
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    db_path = case_dir / "medicare_records.db"
    if db_path.exists():
        db_path.unlink()

    comms_dir = case_dir / "intercepted_comms"
    scans_dir = case_dir / "scanned_claims"
    comms_dir.mkdir(exist_ok=True)
    scans_dir.mkdir(exist_ok=True)

    # Deterministic seeding: combine the case_id with an optional seed so the
    # same (case_id, seed) pair always reproduces byte-for-byte.
    if rng_seed is None:
        digest = hashlib.sha256(case_id.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:8], "big") % (2**31)
    rng = random.Random(rng_seed)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)

    cur.execute("INSERT INTO case_metadata VALUES (?, ?)", ("tier", str(tier)))
    cur.execute("INSERT INTO case_metadata VALUES (?, ?)", ("seed", str(rng_seed)))
    cur.execute("INSERT INTO case_metadata VALUES (?, ?)", ("case_id", case_id))

    corps, bens = _plant_background(cur, rng, tier)
    typologies = _plant_fraud(cur, rng, tier, corps, bens, comms_dir, scans_dir)

    cur.execute("INSERT INTO case_metadata VALUES (?, ?)",
                ("typologies", json.dumps(typologies)))

    conn.commit()
    conn.close()
    return case_dir


# ── Variant entry points (PPP, shell-only, dead-patient) ────────────────────

def generate_ppp_fraud_case(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: int | None = None,
) -> Path:
    """Primary typology: PPP loan fraud (inflated headcount vs payroll records).

    No AKS/dead-patient signals — the only fraud signal is the
    loan_applications ↔ payroll_records contradiction.
    """
    base_dir = Path(base_dir)
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / "medicare_records.db"
    if db_path.exists():
        db_path.unlink()
    comms_dir = case_dir / "intercepted_comms"
    scans_dir = case_dir / "scanned_claims"
    comms_dir.mkdir(exist_ok=True)
    scans_dir.mkdir(exist_ok=True)

    if rng_seed is None:
        digest = hashlib.sha256(case_id.encode()).digest()
        rng_seed = int.from_bytes(digest[:8], "big") % (2 ** 31)
    rng = random.Random(rng_seed)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("tier", str(tier)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("seed", str(rng_seed)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("case_id", case_id))

    corps, bens = _plant_background(cur, rng, tier)
    typologies: list[str] = []

    ppp_eid = _plant_ppp_fraud(cur, rng, typologies)
    cur.execute("INSERT INTO case_metadata VALUES (?,?)",
                ("typologies", json.dumps(typologies)))
    conn.commit()
    conn.close()
    return case_dir


def generate_shell_chain_case(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: int | None = None,
) -> Path:
    """Primary typology: shell-company chain only (4+ layer UBO hierarchy).

    No billing anomalies — the only fraud signal is the link_shell proof chain
    through corporate_registry.
    """
    base_dir = Path(base_dir)
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / "medicare_records.db"
    if db_path.exists():
        db_path.unlink()
    (case_dir / "intercepted_comms").mkdir(exist_ok=True)
    (case_dir / "scanned_claims").mkdir(exist_ok=True)

    if rng_seed is None:
        digest = hashlib.sha256(case_id.encode()).digest()
        rng_seed = int.from_bytes(digest[:8], "big") % (2 ** 31)
    rng = random.Random(rng_seed)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("tier", str(tier)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("seed", str(rng_seed)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("case_id", case_id))

    _plant_background(cur, rng, tier)

    # Plant a 4-layer shell chain in corporate_registry
    fraud_ubo_id = f"U_SHELL_{rng.randint(1000, 9999)}"
    shell_depth = max(4, min(tier + 2, 6))
    prior_eid: str | None = None
    shell_names: list[str] = []
    for layer in range(shell_depth):
        eid = f"E_SHELL_{layer}_{rng.randint(10, 99)}"
        name = (
            f"{rng.choice(['Apex','Vantage','Stratos','Helios','Nexus'])} "
            f"{rng.choice(['Holdings','Capital','Ventures','Partners','Group'])} "
            f"{'LLC' if layer < shell_depth - 1 else 'Corp'}"
        )
        cur.execute("INSERT INTO corporate_registry VALUES (?,?,?,?,?,?,?,?)",
                    (eid, name, f"TX-SH{layer}", prior_eid, fraud_ubo_id,
                     _rand_date(rng, "2019-01-01", "2023-06-01"),
                     rng.choice(_STATES), None))
        cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                    ("entity", json.dumps({"name": name, "kind": "corporation"})))
        if prior_eid is not None:
            cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                        ("shell_link", json.dumps(
                            {"child": name, "parent": shell_names[-1]})))
        shell_names.append(name)
        prior_eid = eid

    cur.execute("INSERT INTO case_metadata VALUES (?,?)",
                ("typologies", json.dumps(["foreign_affiliation"])))
    conn.commit()
    conn.close()
    return case_dir


def generate_dead_patient_case(
    base_dir: Path | str,
    case_id: str,
    tier: int = 1,
    rng_seed: int | None = None,
) -> Path:
    """Primary typology: dead-patient claim.

    beneficiary_summary has BENE_DEATH_DT set before the carrier_claims
    CLM_FROM_DT — single clean contradiction, no AKS noise.
    """
    base_dir = Path(base_dir)
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / "medicare_records.db"
    if db_path.exists():
        db_path.unlink()
    scans_dir = case_dir / "scanned_claims"
    (case_dir / "intercepted_comms").mkdir(exist_ok=True)
    scans_dir.mkdir(exist_ok=True)

    if rng_seed is None:
        digest = hashlib.sha256(case_id.encode()).digest()
        rng_seed = int.from_bytes(digest[:8], "big") % (2 ** 31)
    rng = random.Random(rng_seed)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("tier", str(tier)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("seed", str(rng_seed)))
    cur.execute("INSERT INTO case_metadata VALUES (?,?)", ("case_id", case_id))

    corps, bens = _plant_background(cur, rng, tier)
    typologies: list[str] = []

    # Force at least one beneficiary to have a death date
    bid = f"BENE_DEAD_{rng.randint(1000, 9999)}"
    dob = _rand_date(rng, "1940-01-01", "1960-12-31")
    dod = _rand_date(rng, "2024-01-01", "2025-06-01")
    cur.execute(
        "INSERT INTO beneficiary_summary "
        "(DESYNPUF_ID, BENE_BIRTH_DT, BENE_DEATH_DT, BENE_SEX_IDENT_CD, "
        " BENE_RACE_CD, SP_STATE_CODE) VALUES (?,?,?,?,?,?)",
        (bid, dob, dod, rng.choice(["1", "2"]),
         rng.choice(["1", "2", "3"]), rng.choice(_STATES)),
    )

    fraud_npi = generate_valid_npi(rng)
    bad_date = (datetime.strptime(dod, "%Y-%m-%d")
                + timedelta(days=rng.randint(15, 60))).strftime("%Y-%m-%d")
    cid = f"C_DEAD_{rng.randint(1000, 9999)}"
    cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, bid, bad_date, bad_date, fraud_npi, "TX-F",
                 "99215", 350.0, "428.0"))

    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"beneficiary:{bid}",
                    "evidence_b": f"claim:{cid}",
                    "kind": "dead_patient_claim",
                })))
    typologies.append("dead_patient_claim")

    _emit_pdf(cur, scans_dir, tier, rng, ClaimEvidence(
        claim_id=cid, beneficiary_id=bid, beneficiary_dob=dob,
        beneficiary_dod=dod, provider_name="Unknown Provider",
        provider_npi=fraud_npi, service_date=bad_date,
        hcpcs_code="99215", icd9_code="428.0", amount=350.0,
        diagnosis_text="Post-mortem billing — congestive heart failure",
    ))

    cur.execute("INSERT INTO case_metadata VALUES (?,?)",
                ("typologies", json.dumps(typologies)))
    conn.commit()
    conn.close()
    return case_dir


# ── Background (legitimate) data ─────────────────────────────────────────────

def _plant_background(cur, rng: random.Random, tier: int):
    n_legit_corps = 5 + tier * 3
    n_bens = 20 + tier * 10
    n_carrier = 80 + tier * 40
    n_pde_providers = 12 + tier * 4

    corps: list[tuple[str, str, str]] = []
    for i in range(n_legit_corps):
        eid = f"E_L{i:03d}"
        name = (f"{rng.choice(['Legit','Professional','Mercy','Main Street','Harbor'])} "
                f"{rng.choice(['Medical','Health','Clinical','Wellness'])} "
                f"{rng.choice(['LLC','Corp','Inc','Ltd'])} {i}")
        state = rng.choice(_STATES)
        npi = generate_valid_npi(rng)
        cur.execute("INSERT INTO corporate_registry VALUES (?,?,?,?,?,?,?,?)",
                    (eid, name, f"TX-{1000+i}", None, f"U_L{i:03d}",
                     _rand_date(rng, "2010-01-01", "2023-12-31"), state, npi))
        corps.append((eid, name, npi))

    bens: list[tuple[str, str, str | None]] = []
    for i in range(n_bens):
        bid = f"BENE_{i:04d}"
        dob = _rand_date(rng, "1940-01-01", "1960-12-31")
        dod = _rand_date(rng, "2024-01-01", "2026-04-01") if rng.random() < 0.10 else None
        cur.execute(
            "INSERT INTO beneficiary_summary "
            "(DESYNPUF_ID, BENE_BIRTH_DT, BENE_DEATH_DT, BENE_SEX_IDENT_CD, "
            " BENE_RACE_CD, SP_STATE_CODE) VALUES (?,?,?,?,?,?)",
            (bid, dob, dod, rng.choice(["1", "2"]),
             rng.choice(["1", "2", "3"]), rng.choice(_STATES)),
        )
        bens.append((bid, dob, dod))

    # Carrier claims (legitimate volume)
    for i in range(n_carrier):
        cid = f"C_L{i:05d}"
        bid, _, dod = rng.choice(bens)
        _, _, npi = rng.choice(corps)
        sdate = _rand_date(rng, "2024-01-01", "2026-03-01")
        if dod and sdate > dod:
            sdate = (datetime.strptime(dod, "%Y-%m-%d")
                     - timedelta(days=rng.randint(5, 100))).strftime("%Y-%m-%d")
        amt = _benford_amount(rng, 80, 1500)
        cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, bid, sdate, sdate, npi, "TX-P",
                     rng.choice(_HCPCS_CODES), amt, rng.choice(_ICD9_CODES)))

    # PDE background noise — many providers, few scripts each, common drugs.
    legit_prescriber_npis = [generate_valid_npi(rng) for _ in range(n_pde_providers)]
    pde_counter = 0
    for npi in legit_prescriber_npis:
        for _ in range(rng.randint(5, 15)):
            pde_id = f"PDE_L{pde_counter:06d}"
            pde_counter += 1
            bid, _, _ = rng.choice(bens)
            cur.execute("INSERT INTO prescription_drug_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (pde_id, bid, npi,
                         _rand_date(rng, "2025-01-01", "2025-12-31"),
                         f"NDC-{rng.randint(10000,99999)}",
                         rng.choice(_LEGIT_DRUGS),
                         rng.randint(30, 90), rng.randint(30, 90),
                         round(rng.uniform(2.0, 15.0), 2),
                         round(rng.uniform(10.0, 60.0), 2)))

    return corps, bens


# ── Fraud planting ───────────────────────────────────────────────────────────

def _emit_pdf(
    cur,
    scans_dir: Path,
    tier: int,
    rng: random.Random,
    ev: ClaimEvidence,
) -> str:
    """Render the CMS-1500 and log to evidence_documents. Returns pdf path."""
    pdf_path = scans_dir / f"{ev.claim_id}.pdf"
    expected = render_claim_pdf(pdf_path, ev, tier=tier, rng=rng)
    # Store a path relative to the case directory so the agent (whose CWD is
    # the sandbox'd case dir) can open it with a stable string.
    rel_path = str(pdf_path.relative_to(scans_dir.parent)).replace(os.sep, "/")
    cur.execute(
        "INSERT INTO evidence_documents VALUES (?,?,?,?,?,?)",
        (f"DOC_{ev.claim_id}", ev.claim_id, rel_path, tier, 1,
         json.dumps(expected)),
    )
    return rel_path


def _plant_fraud(
    cur,
    rng: random.Random,
    tier: int,
    corps: list[tuple[str, str, str]],
    bens: list[tuple[str, str, str | None]],
    comms_dir: Path,
    scans_dir: Path,
) -> list[str]:
    typologies: list[str] = []

    # ── 1. Shell-company chain (tier-scaled depth) ──────────────────────────
    fraud_ubo_id = f"U_FRAUD_{rng.randint(1000, 9999)}"
    shell_depth = max(2, min(tier + 1, 5))  # tier 1 → 2 shells, tier 5 → 5 shells
    shell_names: list[str] = []
    prior_eid: str | None = None
    fraud_entity_npi: str | None = None
    for layer in range(shell_depth):
        eid = f"E_FRAUD_{layer}"
        name = (f"{rng.choice(['Shadow','Phantom','Ghost','Dark','Apex','Helix'])} "
                f"{rng.choice(['Operations','Medical','Consulting','Research','Advisors'])} "
                f"{'LLC' if layer < shell_depth - 1 else 'Corp'}")
        state = rng.choice(_STATES)
        is_terminal = layer == shell_depth - 1
        npi = generate_valid_npi(rng) if is_terminal else None
        if is_terminal:
            fraud_entity_npi = npi
        cur.execute("INSERT INTO corporate_registry VALUES (?,?,?,?,?,?,?,?)",
                    (eid, name, f"TX-F{layer}", prior_eid, fraud_ubo_id,
                     _rand_date(rng, "2020-01-01", "2023-01-01"), state, npi))
        cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                    ("entity", json.dumps({"name": name, "kind": "corporation"})))
        if prior_eid is not None:
            cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                        ("shell_link", json.dumps(
                            {"child": name, "parent": shell_names[-1]})))
        shell_names.append(name)
        prior_eid = eid

    # ── 2. Fraud provider (the doctor) ──────────────────────────────────────
    fraud_prov_name = rng.choice(_FRAUD_DOCTORS)
    fraud_npi = generate_valid_npi(rng)
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("entity", json.dumps(
                    {"name": fraud_prov_name, "kind": "provider", "npi": fraud_npi})))
    # Also record the doctor in corporate_registry so CaseHandle.all_entity_names()
    # returns them and the CoT-grounding check can recognise the name.
    cur.execute("INSERT INTO corporate_registry VALUES (?,?,?,?,?,?,?,?)",
                (f"E_PROV_FRAUD", fraud_prov_name, "TX-PROV", None, f"U_P_FRAUD",
                 _rand_date(rng, "2018-01-01", "2022-01-01"),
                 rng.choice(_STATES), fraud_npi))

    # ── 3. AKS kickback: PDE spike + smoking gun email + referral payment ──
    shell_company = shell_names[-1]  # deepest shell is the "pharma front"
    target_drug = rng.choice(_SCRUTINIZED_DRUGS)
    anomalous_scripts = 150 + tier * 50
    grant_amount = 25000.0 + tier * 5000.0

    planted_pde_ids: list[str] = []
    for i in range(anomalous_scripts):
        pde_id = f"PDE_FRAUD_{i:06d}"
        planted_pde_ids.append(pde_id)
        bid, _, _ = rng.choice(bens)
        cur.execute("INSERT INTO prescription_drug_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pde_id, bid, fraud_npi,
                     _rand_date(rng, "2025-02-01", "2025-11-30"),
                     f"NDC-{rng.randint(10000,99999)}",
                     target_drug, rng.randint(30, 120), rng.randint(30, 120),
                     round(rng.uniform(50.0, 150.0), 2),
                     round(rng.uniform(500.0, 1200.0), 2)))

    # Ledger + referral payment trail for the "research grant"
    payment_date = _rand_date(rng, "2025-01-05", "2025-01-30")
    tx_id = f"TX_KICKBACK_{rng.randint(100000, 999999)}"
    cur.execute("INSERT INTO general_ledger VALUES (?,?,?,?,?,?,?)",
                (tx_id, payment_date, "Consulting Expense", "Cash",
                 grant_amount, f"Research grant to {fraud_prov_name}",
                 f"E_FRAUD_{shell_depth - 1}"))
    cur.execute("INSERT INTO referral_payments VALUES (?,?,?,?,?,?)",
                (f"RP_{rng.randint(100000,999999)}",
                 fraud_entity_npi or "UNKNOWN", fraud_npi, grant_amount,
                 payment_date, "speaker_fee / research_grant"))

    # Benign email haystack + smoking gun (OCR-noised NPI)
    smoking_gun_idx = rng.randint(10, 40)
    for i in range(50):
        fp = comms_dir / f"email_{i:02d}.txt"
        if i == smoking_gun_idx:
            noisy_npi = _inject_ocr_noise(fraud_npi, rng, n_errors=2)
            body = (
                "CONFIDENTIAL — DO NOT FORWARD\n"
                f"To: {shell_company} Disbursals\n"
                f"From: Regional VP of Sales\n"
                f"Date: {payment_date}\n\n"
                f"Authorize 'Research Grant' payment of ${grant_amount:,.0f} "
                f"to {fraud_prov_name}.\n"
                f"Since the payment cleared last month, his {target_drug} "
                "script volume is up 400%.\n"
                "Make sure the paperwork looks like a legitimate consulting fee.\n\n"
                f"Provider Identifier for records: NPl {noisy_npi}\n"
            )
            fp.write_text(body, encoding="utf-8")
        else:
            fp.write_text(
                f"Subject: Weekly update\n\n"
                f"Nothing to report for week {i}. Regards, Compliance.\n",
                encoding="utf-8",
            )
    smoking_gun_path = f"intercepted_comms/email_{smoking_gun_idx:02d}.txt"

    # Also emit a degraded CMS-1500 PDF for one of the fraudulent PDE claims,
    # so the agent has a multi-modal thread (PDE → PDF claim → email).
    # Pick any beneficiary — tie the PDF to a synthetic carrier_claim so the
    # evidence_documents FK resolves.
    pdf_bid, pdf_dob, _ = rng.choice(bens)
    pdf_cid = "C_FRAUD_AKS_PDF"
    pdf_service_date = _rand_date(rng, "2025-03-01", "2025-10-01")
    pdf_amount = round(rng.uniform(500.0, 1200.0), 2)
    cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                (pdf_cid, pdf_bid, pdf_service_date, pdf_service_date, fraud_npi,
                 "TX-F", "J0171", pdf_amount, "401.9"))
    pdf_rel = _emit_pdf(cur, scans_dir, tier, rng, ClaimEvidence(
        claim_id=pdf_cid, beneficiary_id=pdf_bid, beneficiary_dob=pdf_dob,
        beneficiary_dod=None, provider_name=fraud_prov_name, provider_npi=fraud_npi,
        service_date=pdf_service_date, hcpcs_code="J0171", icd9_code="401.9",
        amount=pdf_amount, diagnosis_text="Chronic pain management follow-up",
    ))

    # Ground truth: AKS contradiction + rich payload so the grader can verify
    # either the email or the PDE spike as primary evidence.
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": smoking_gun_path,
                    "evidence_b": f"provider_npi:{fraud_npi}",
                    "kind": "aks_violation",
                    "fraud_npi": fraud_npi,
                    "shell_company": shell_company,
                    "target_drug": target_drug,
                    "pdf_evidence": pdf_rel,
                    "grant_amount": grant_amount,
                })))
    typologies.append("aks_violation")

    # ── 4. Dead patient claim ───────────────────────────────────────────────
    dead_bens = [b for b in bens if b[2] is not None]
    if dead_bens:
        bid, dob, dod = dead_bens[0]
        bad_date = (datetime.strptime(dod, "%Y-%m-%d")
                    + timedelta(days=rng.randint(20, 60))).strftime("%Y-%m-%d")
        cid = "C_FRAUD_DEAD"
        cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, bid, bad_date, bad_date, fraud_npi, "TX-F",
                     "99215", 350.0, "428.0"))
        cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                    ("contradiction", json.dumps({
                        "evidence_a": f"beneficiary:{bid}",
                        "evidence_b": f"claim:{cid}",
                        "kind": "dead_patient_claim"})))
        _emit_pdf(cur, scans_dir, tier, rng, ClaimEvidence(
            claim_id=cid, beneficiary_id=bid, beneficiary_dob=dob,
            beneficiary_dod=dod, provider_name=fraud_prov_name,
            provider_npi=fraud_npi, service_date=bad_date,
            hcpcs_code="99215", icd9_code="428.0", amount=350.0,
            diagnosis_text="Congestive heart failure — follow-up",
        ))
        typologies.append("dead_patient_claim")

    # ── 5. Duplicate billing ────────────────────────────────────────────────
    bid_dup, dob_dup, _ = bens[0]
    dup_date = "2025-10-10"
    for suffix in ["A", "B"]:
        cid = f"C_FRAUD_DUP_{suffix}"
        cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, bid_dup, dup_date, dup_date, fraud_npi, "TX-F",
                     "99214", 200.0, "401.1"))
        _emit_pdf(cur, scans_dir, tier, rng, ClaimEvidence(
            claim_id=cid, beneficiary_id=bid_dup, beneficiary_dob=dob_dup,
            beneficiary_dod=None, provider_name=fraud_prov_name,
            provider_npi=fraud_npi, service_date=dup_date,
            hcpcs_code="99214", icd9_code="401.1", amount=200.0,
            diagnosis_text="Essential hypertension, unspecified",
        ))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": "claim:C_FRAUD_DUP_A",
                    "evidence_b": "claim:C_FRAUD_DUP_B",
                    "kind": "duplicate_bill"})))
    typologies.append("duplicate_bill")

    # ── 6. Upcoding (tier ≥ 2) — PDF narrative contradicts HCPCS ────────────
    if tier >= 2 and len(bens) > 1:
        bid_up, dob_up, _ = bens[1]
        cid = "C_FRAUD_UPCODE"
        cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, bid_up, "2025-12-01", "2025-12-01", fraud_npi, "TX-F",
                     "99215", 850.0, "401.1"))
        cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                    ("contradiction", json.dumps({
                        "evidence_a": f"claim:{cid}",
                        "evidence_b": f"provider_npi:{fraud_npi}",
                        "kind": "upcoding"})))
        _emit_pdf(cur, scans_dir, tier, rng, ClaimEvidence(
            claim_id=cid, beneficiary_id=bid_up, beneficiary_dob=dob_up,
            beneficiary_dod=None, provider_name=fraud_prov_name,
            provider_npi=fraud_npi, service_date="2025-12-01",
            hcpcs_code="99215", icd9_code="401.1", amount=850.0,
            diagnosis_text="Routine BP check. No acute issues. Patient stable.",
        ))
        typologies.append("upcoding")

    # ── Tier-gated additional typologies ────────────────────────────────────
    if tier >= 2:
        _plant_unbundling(cur, rng, bens, fraud_npi, typologies)

    if tier >= 3:
        _plant_phantom_beneficiary(cur, rng, bens, fraud_npi, typologies)
        _plant_off_label_marketing(cur, rng, bens, fraud_npi, typologies)

    if tier >= 4:
        contractor_a = _plant_double_billing(cur, rng, typologies)
        contractor_b = _plant_cost_pricing_fraud(cur, rng, typologies)

    if tier >= 5:
        contractor_c = _plant_product_substitution(cur, rng, typologies)
        contractor_d = _plant_ppp_fraud(cur, rng, typologies)
        # Foreign affiliation attaches to one of the gov-contracting fraud entities,
        # so the agent has to cross domains to discover it.
        _plant_foreign_affiliation(cur, rng, contractor_d, typologies)

    return typologies


# ── Per-typology helpers (tier 2+) ────────────────────────────────────────────

def _register_contractor(cur, rng: random.Random, role: str) -> tuple[str, str]:
    """Register a fraud contractor in corporate_registry. Returns (entity_id, name)."""
    eid = f"E_FRAUD_CONTRACT_{role}_{rng.randint(1000, 9999)}"
    name = rng.choice(_FRAUD_CONTRACTORS) + f" #{rng.randint(10, 99)}"
    state = rng.choice(_STATES)
    cur.execute("INSERT INTO corporate_registry VALUES (?,?,?,?,?,?,?,?)",
                (eid, name, f"TX-CTR-{role}", None, f"U_CTR_{role}",
                 _rand_date(rng, "2015-01-01", "2022-12-31"), state, None))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("entity", json.dumps({"name": name, "kind": "contractor"})))
    return eid, name


def _plant_unbundling(cur, rng, bens, fraud_npi: str, typologies: list[str]) -> None:
    """Bundle that should have been billed as one code is split into components."""
    bundle_code, components = rng.choice(_UNBUNDLING_PAIRS)
    bid, _, _ = rng.choice(bens)
    sdate = "2025-09-12"
    component_cids: list[str] = []
    for i, comp in enumerate(components):
        cid = f"C_FRAUD_UNBUND_{i}"
        component_cids.append(cid)
        cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, bid, sdate, sdate, fraud_npi, "TX-F",
                     comp, _benford_amount(rng, 60, 200), "401.1"))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"claim:{component_cids[0]}",
                    "evidence_b": f"claim:{component_cids[-1]}",
                    "kind": "unbundling",
                    "bundle_code": bundle_code,
                    "components": components,
                })))
    typologies.append("unbundling")


def _plant_phantom_beneficiary(cur, rng, bens, fraud_npi: str,
                               typologies: list[str]) -> None:
    """Claim filed for a beneficiary_id that does not exist in beneficiary_summary."""
    phantom_bid = f"BENE_PHANTOM_{rng.randint(10000, 99999)}"
    cid = "C_FRAUD_PHANTOM"
    sdate = _rand_date(rng, "2025-04-01", "2025-11-30")
    cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, phantom_bid, sdate, sdate, fraud_npi, "TX-F",
                 "99214", _benford_amount(rng, 200, 600), "250.0"))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"claim:{cid}",
                    "evidence_b": f"beneficiary:{phantom_bid}",
                    "kind": "phantom_beneficiary",
                    "phantom_bid": phantom_bid,
                })))
    typologies.append("phantom_beneficiary")


def _plant_off_label_marketing(cur, rng, bens, fraud_npi: str,
                               typologies: list[str]) -> None:
    """Drug prescribed for a diagnosis outside its FDA-approved indication."""
    drug = rng.choice(list(_OFFLABEL_DRUGS.keys()))
    approved = set(_OFFLABEL_DRUGS[drug]["approved_icd"])
    off_label_icd = next((c for c in _ICD9_CODES if c not in approved), "401.1")
    bid, _, _ = rng.choice(bens)
    sdate = _rand_date(rng, "2025-05-01", "2025-11-30")
    pde_id = f"PDE_FRAUD_OFFLABEL_{rng.randint(10000, 99999)}"
    cur.execute("INSERT INTO prescription_drug_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pde_id, bid, fraud_npi, sdate,
                 f"NDC-{rng.randint(10000, 99999)}", drug,
                 rng.randint(30, 90), rng.randint(30, 90),
                 round(rng.uniform(50.0, 200.0), 2),
                 round(rng.uniform(800.0, 2000.0), 2)))
    cid = "C_FRAUD_OFFLABEL"
    cur.execute("INSERT INTO carrier_claims VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, bid, sdate, sdate, fraud_npi, "TX-F",
                 "99214", _benford_amount(rng, 100, 400), off_label_icd))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"pde:{pde_id}",
                    "evidence_b": f"claim:{cid}",
                    "kind": "off_label_marketing",
                    "drug": drug,
                    "approved_icd": list(approved),
                    "billed_icd": off_label_icd,
                })))
    typologies.append("off_label_marketing")


def _plant_double_billing(cur, rng, typologies: list[str]) -> str:
    """Same line-item invoiced twice against one government contract."""
    eid, _ = _register_contractor(cur, rng, "DBL")
    contract_id = f"CTR_DBL_{rng.randint(1000, 9999)}"
    line = "Helmet liner, qty 50"
    amt = round(rng.uniform(15000, 40000), 2)
    cur.execute("INSERT INTO government_contracts VALUES (?,?,?,?,?,?,?,?)",
                (contract_id, rng.choice(_AGENCIES), eid, amt * 2,
                 _rand_date(rng, "2024-06-01", "2024-12-31"),
                 "Helmet liner", round(amt / 50, 2), round(amt / 50, 2)))
    inv_a = f"INV_{contract_id}_A"
    inv_b = f"INV_{contract_id}_B"
    same_date = _rand_date(rng, "2025-01-15", "2025-03-15")
    cur.execute("INSERT INTO contract_invoices VALUES (?,?,?,?,?)",
                (inv_a, contract_id, line, amt, same_date))
    cur.execute("INSERT INTO contract_invoices VALUES (?,?,?,?,?)",
                (inv_b, contract_id, line, amt, same_date))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"invoice:{inv_a}",
                    "evidence_b": f"invoice:{inv_b}",
                    "kind": "double_billing",
                    "contract_id": contract_id,
                })))
    typologies.append("double_billing")
    return eid


def _plant_cost_pricing_fraud(cur, rng, typologies: list[str]) -> str:
    """Disclosed unit price >> actual unit cost on a cost-plus government contract."""
    eid, _ = _register_contractor(cur, rng, "CPF")
    contract_id = f"CTR_CPF_{rng.randint(1000, 9999)}"
    actual_cost = round(rng.uniform(50, 200), 2)
    disclosed = round(actual_cost * rng.uniform(2.5, 4.5), 2)  # gross markup
    cur.execute("INSERT INTO government_contracts VALUES (?,?,?,?,?,?,?,?)",
                (contract_id, rng.choice(_AGENCIES), eid,
                 disclosed * rng.randint(500, 2000),
                 _rand_date(rng, "2024-03-01", "2024-09-30"),
                 "Field maintenance kit", disclosed, actual_cost))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"contract:{contract_id}:disclosed_unit_price",
                    "evidence_b": f"contract:{contract_id}:actual_unit_cost",
                    "kind": "cost_pricing_fraud",
                    "contract_id": contract_id,
                    "markup": round(disclosed / max(actual_cost, 0.01), 2),
                })))
    typologies.append("cost_pricing_fraud")
    return eid


def _plant_product_substitution(cur, rng, typologies: list[str]) -> str:
    """Delivered product differs from the one specified in the contract."""
    eid, _ = _register_contractor(cur, rng, "SUB")
    contract_id = f"CTR_SUB_{rng.randint(1000, 9999)}"
    expected = rng.choice(_PRODUCTS_EXPECTED)
    substituted = rng.choice(_PRODUCTS_SUBSTITUTED)
    cur.execute("INSERT INTO government_contracts VALUES (?,?,?,?,?,?,?,?)",
                (contract_id, rng.choice(_AGENCIES), eid,
                 round(rng.uniform(200000, 800000), 2),
                 _rand_date(rng, "2024-02-01", "2024-08-31"),
                 expected, round(rng.uniform(800, 1500), 2),
                 round(rng.uniform(200, 600), 2)))
    delivery_id = f"DLV_{contract_id}"
    cur.execute("INSERT INTO contract_deliveries VALUES (?,?,?,?)",
                (delivery_id, contract_id, substituted,
                 _rand_date(rng, "2024-09-01", "2025-02-28")))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"contract:{contract_id}:expected_product",
                    "evidence_b": f"delivery:{delivery_id}:delivered_product",
                    "kind": "product_substitution",
                    "expected": expected,
                    "delivered": substituted,
                })))
    typologies.append("product_substitution")
    return eid


def _plant_ppp_fraud(cur, rng, typologies: list[str]) -> str:
    """PPP loan claims an employee count well above what payroll records show."""
    eid, _ = _register_contractor(cur, rng, "PPP")
    loan_id = f"PPP_{rng.randint(100000, 999999)}"
    real_employees = rng.randint(3, 15)
    claimed = real_employees + rng.randint(40, 120)
    claimed_payroll = claimed * rng.uniform(4500, 7000)
    loan_amt = round(claimed_payroll * 2.5, 2)
    app_date = _rand_date(rng, "2020-04-15", "2021-05-31")
    cur.execute("INSERT INTO loan_applications VALUES (?,?,?,?,?,?,?)",
                (loan_id, eid, "PPP", claimed, round(claimed_payroll, 2),
                 loan_amt, app_date))
    # Plant 3 quarters of payroll records that contradict the claimed count.
    for q in range(3):
        period_end = _rand_date(rng, "2020-03-31", "2021-09-30")
        cur.execute("INSERT INTO payroll_records VALUES (?,?,?,?,?)",
                    (f"PAY_{loan_id}_{q}", eid, period_end, real_employees,
                     round(real_employees * rng.uniform(4500, 7000), 2)))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"loan:{loan_id}",
                    "evidence_b": f"payroll:{eid}",
                    "kind": "ppp_fraud",
                    "claimed_employees": claimed,
                    "actual_employees": real_employees,
                })))
    typologies.append("ppp_fraud")
    return eid


def _plant_foreign_affiliation(cur, rng, contractor_eid: str,
                               typologies: list[str]) -> None:
    """Undisclosed foreign parent for a domestic vendor."""
    foreign_parent, country = rng.choice(_FOREIGN_PARENTS)
    cur.execute("INSERT INTO foreign_affiliations VALUES (?,?,?,?)",
                (contractor_eid, foreign_parent, country, "undisclosed"))
    cur.execute("INSERT INTO ground_truth VALUES (?,?)",
                ("contradiction", json.dumps({
                    "evidence_a": f"entity:{contractor_eid}",
                    "evidence_b": f"foreign_parent:{foreign_parent}",
                    "kind": "foreign_affiliation",
                    "country": country,
                })))
    typologies.append("foreign_affiliation")


# ── CLI removed ──────────────────────────────────────────────────────────────
# The case-bank generator CLI lives in `data_gen/build_case_bank.py`.
# This module exposes only `generate_multimodal_aks_case()` for programmatic use
# (the environment imports it directly for on-the-fly fallback generation).

