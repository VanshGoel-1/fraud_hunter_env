# AGENTS.md

Project-level instructions for Codex sessions inside this repo.

## What this is

`fraud_hunter_env/` is the project root — an **OpenEnv-compliant, 7-layer graded RL environment** for the Government Fraud Hunter AI hackathon. The parent `hackathon/` directory is just a container; treat **this directory** as the root for all paths, imports, and deploy commands.

The environment implements a full fraud-investigation loop: agents query synthetic Medicare + corporate-registry databases, extract entities, link shell companies, OCR scanned claim PDFs, read intercepted communications, and submit a case file for programmatic grading. The grader applies seven hierarchical reward layers (format gate, CoT enforcement, CoT grounding, length schedule, duplicate detection, NPI Luhn validation, per-typology reward matrix) with a proof-chain bonus.

## Directory layout

```
fraud_hunter_env/              # project root (this directory)
├── __init__.py                # package exports
├── client.py                  # FraudHunterEnv(EnvClient) — WebSocket + HTTP client
├── models.py                  # Pydantic Action/Observation schemas + reward constants
├── schema.py                  # SQL table names, typology→source map (single source of truth)
├── npi_utils.py               # Luhn-valid NPI generation and validation
├── replay.py                  # SHA-256 replay chain verifier
├── config.py                  # env-var config helpers
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # project metadata + deps (uv-managed)
├── uv.lock                    # locked deps
├── eval.py                    # baseline-vs-expert evaluation harness (HTTP/WS client)
├── inference.py               # smoke-test script over the network client
├── demo.py                    # interactive session demo
├── blog.md                    # technical writeup
├── build_case_bank.py         # convenience wrapper for data_gen.build_case_bank
├── README.md                  # project README
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI + WebSocket (/ws, /reset, /step, /state, /metrics, ...)
    ├── fraud_hunter_env_environment.py  # FraudHunterEnvironment — reset/step/state (RLVE + RLVR)
    ├── grader.py              # 7-layer RLVR grader
    ├── difficulty.py          # 5-tier RLVE DifficultyManager
    ├── data_loader.py         # tiered SQLite case loader + demo case
    ├── sandbox.py             # CodeAct Python / SQL sandbox (5s timeout, read-only DB)
    ├── online_rl.py           # lightweight contextual bandit for real-time arm selection
    ├── metrics_bus.py         # in-memory SSE fan-out + leaderboard
    ├── http_contract.py       # shared HTTP/TestClient helpers
    └── Dockerfile             # container image (multi-stage, uv sync)
├── data_gen/
    ├── case_compiler.py       # CMS SynPUF synthetic case generator (multi-modal AKS cases)
    ├── typology_dispatcher.py # tier-aware variant selector (PPP, shell, dead-patient, AKS)
    ├── build_case_bank.py     # batch case bank builder (test: 10/tier, production: 2000/tier)
    └── pdf_evidence.py        # CMS-1500-style degraded PDF renderer (Pillow)
├── training/
    ├── grpo_train.ipynb       # Colab-ready GRPO + DAPO training notebook
    └── (grpo_train.py deleted — use the notebook)
├── tests/
    ├── test_case_contracts.py
    ├── test_environment.py
    ├── test_grader.py
    ├── test_http_actions.py
    ├── test_models.py
    ├── test_online_rl.py
    ├── test_replay.py
    ├── test_typology_dispatcher.py
    ├── test_upload_and_nl_api.py
    └── smoke/
        └── test_end_to_end_smoke.py
├── scripts/
    ├── validate_runtime.py    # fast compile + HTTP surface check
    └── http_surface_check.py  # in-process action surface validator
├── web/index.html             # live SSE monitoring dashboard
└── assets/                    # generated plots + eval_results.json
```

Deploy command: `openenv push` from this directory (where `openenv.yaml` lives).

## Session rules

1. **Do not rewrite the generator output unprompted.** Files produced by `openenv init`
   (`client.py`, `models.py`, `server/*.py`, `Dockerfile`, `openenv.yaml`, `pyproject.toml`,
   `README.md`) are the canonical scaffold. Edit them only when the user explicitly asks
   or when implementing a concrete feature — do not add TODOs, rename files, flip
   constants, or "polish" unprompted.
2. **Ignore `docs/`.** The `docs/` subfolder contains older design documents (prd.md,
   trd.md, implementation_plan.md, SYSTEM.md, STATE.yaml, CHANGELOG.md, etc.) from
   prior runs. They're rudimentary and not authoritative — do not read, cite, or
   update them unless the user explicitly asks. Work from user direction and the
   generator output.
3. **Protocol invariant.** The WebSocket endpoint `/ws` is the authoritative protocol.
   HTTP `/step` and `/reset` are convenience endpoints for local development only.
4. **Before risky actions** (deleting files, renaming many files, pushing to HF,
   force-push, nuking state), confirm with the user first.

## Conventions

- Package name is `fraud_hunter_env`; keep module paths consistent with what `openenv
  init` generated.
- Pydantic 2 schemas live in `models.py` and only there — no logic, no I/O.
- `server/fraud_hunter_env_environment.py` is where `reset()`, `step()`, and the `state`
  property live. Keep the generator's filename; do not rename without a matching
  import update everywhere.
- Tests go under `tests/` at this directory, not nested in `server/`.
- **Canonical case-bank schema is CMS SynPUF naming** (`beneficiary_summary`, `carrier_claims`,
  `inpatient_claims`, `outpatient_claims`, `prescription_drug_events`, `corporate_registry`,
  `general_ledger`, `referral_payments`, `evidence_documents`, `ground_truth`, `case_metadata`).
  This is what `data_gen/case_compiler.py` produces and what `server/grader.py` queries —
  do not introduce parallel `medicare_*` / `providers` table names.

## External references

- Upstream OpenEnv framework: <https://github.com/meta-pytorch/OpenEnv> — authoritative
  for the Gym-style API (`reset`, `step`, `state`) and the `openenv init` / `openenv push`
  CLI.
- NotebookLM "The Future of AI and Industrial Innovation" (library id
  `the-future-of-ai-and-industria`) is registered in the `notebooklm` MCP server. Consult
  it before guessing on OpenEnv / vLLM MRV2 / GRPO / HF Spaces specifics.
