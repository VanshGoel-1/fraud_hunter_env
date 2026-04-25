"""
Single source of truth for case-database schema knowledge.

Eliminates the previous drift hazard where the environment's table-tracking
list (`step()`) and the grader's per-typology source map (`compute_agentic_recall`)
duplicated the same knowledge in two places.

Anything that needs to *know* what tables/sources exist or which sources back
which fraud typology MUST import from this module — no string literals scattered
through the codebase.
"""

from __future__ import annotations

# ── SQL tables present in every generated case database ─────────────────────

SQL_TABLES: tuple[str, ...] = (
    # CMS SynPUF — claims & beneficiaries
    "beneficiary_summary",
    "carrier_claims",
    "inpatient_claims",
    "outpatient_claims",
    "prescription_drug_events",
    # Corporate / forensic accounting
    "corporate_registry",
    "general_ledger",
    "referral_payments",
    # Government contracting
    "government_contracts",
    "contract_invoices",
    "contract_deliveries",
    # PPP / pandemic
    "loan_applications",
    "payroll_records",
    "foreign_affiliations",
    # Multi-modal evidence index
    "evidence_documents",
)

# ── Off-database evidence directories under each case dir ───────────────────

FILESYSTEM_EVIDENCE_DIRS: tuple[str, ...] = (
    "intercepted_comms",
    "scanned_claims",
)

# Aggregate set used by the environment to track which sources the agent has
# touched (for agentic-recall scoring).
ALL_QUERYABLE_SOURCES: frozenset[str] = frozenset(SQL_TABLES + FILESYSTEM_EVIDENCE_DIRS)


# ── Per-typology source map ─────────────────────────────────────────────────
# For each fraud typology, the set of sources the agent SHOULD consult to
# uncover that typology. Used by `compute_agentic_recall`.

TYPOLOGY_SOURCES: dict[str, frozenset[str]] = {
    "aks_violation": frozenset({
        "prescription_drug_events", "general_ledger", "referral_payments",
        "intercepted_comms", "scanned_claims", "evidence_documents",
    }),
    "dead_patient_claim": frozenset({
        "beneficiary_summary", "carrier_claims",
        "scanned_claims", "evidence_documents",
    }),
    "duplicate_bill": frozenset({"carrier_claims", "evidence_documents"}),
    "upcoding": frozenset({
        "carrier_claims", "scanned_claims", "evidence_documents",
    }),
    "unbundling": frozenset({"carrier_claims"}),
    "phantom_beneficiary": frozenset({"carrier_claims", "beneficiary_summary"}),
    "off_label_marketing": frozenset({
        "prescription_drug_events", "carrier_claims",
    }),
    "double_billing": frozenset({"government_contracts", "contract_invoices"}),
    "cost_pricing_fraud": frozenset({"government_contracts"}),
    "product_substitution": frozenset({
        "government_contracts", "contract_deliveries",
    }),
    "ppp_fraud": frozenset({"loan_applications", "payroll_records"}),
    "foreign_affiliation": frozenset({
        "foreign_affiliations", "corporate_registry",
    }),
}

# Ground-truth-kind → source set (used when typology isn't in payload)
GT_KIND_SOURCES: dict[str, frozenset[str]] = {
    "entity": frozenset({"corporate_registry", "beneficiary_summary"}),
    "shell_link": frozenset({"corporate_registry"}),
}


__all__ = [
    "SQL_TABLES",
    "FILESYSTEM_EVIDENCE_DIRS",
    "ALL_QUERYABLE_SOURCES",
    "TYPOLOGY_SOURCES",
    "GT_KIND_SOURCES",
]
