# AGENTS.md

Project-level instructions for Codex sessions inside this repo.

## What this is

`fraud_hunter_env/` is the project root — an **OpenEnv-compliant RL environment** for the
Government Fraud Hunter AI hackathon. The parent `hackathon/` directory is just a
container; treat **this directory** as the root for all paths, imports, and deploy commands.

The scaffold was produced by `openenv init fraud_hunter_env` and currently implements a
**placeholder echo environment**. The real job is to replace that with a fraud-investigation
environment: agents query simulated Medicare + corporate-registry databases, extract
entities, link shell companies, and submit a case file for programmatic grading.

## Directory layout

```
fraud_hunter_env/              # project root (this directory)
├── __init__.py                # package exports
├── client.py                  # FraudHunterEnv(EnvClient) — WebSocket client
├── models.py                  # Pydantic Action/Observation — echo placeholder
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # project metadata + deps
├── uv.lock                    # locked deps
├── README.md                  # HF Space card + usage
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI + WebSocket (/reset /step /state /ws)
    ├── fraud_hunter_env_environment.py  # Environment(reset/step/state) — echo placeholder
    ├── Dockerfile             # container image
    └── requirements.txt       # pip fallback
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
3. **Protocol invariant.** On Hugging Face Spaces the WebSocket endpoint `/ws` is the
   authoritative protocol. HTTP `/step` and `/reset` are convenience endpoints for local
   development only.
4. **Before risky actions** (deleting files, renaming many files, pushing to HF,
   force-push, nuking state), confirm with the user first.

## Conventions

- Package name is `fraud_hunter_env`; keep module paths consistent with what `openenv
  init` generated.
- Pydantic 2 schemas live in `models.py` and only there — no logic, no I/O.
- `server/fraud_hunter_env_environment.py` is where `reset()`, `step()`, and the `state`
  property live. Keep the generator's filename; do not rename without a matching
  import update everywhere.
- Tests (when added) go under `tests/` at this directory, not nested in `server/`.
- **Canonical case-bank schema is CMS SynPUF naming** (`beneficiary_summary`, `carrier_claims`,
  `inpatient_claims`, `outpatient_claims`, `prescription_drug_events`, `corporate_registry`,
  `general_ledger`, `referral_payments`, `evidence_documents`, `ground_truth`, `case_metadata`).
  This is what `data_gen/case_compiler.py` produces and what `server/grader.py` queries —
  do not introduce parallel `medicare_*` / `providers` table names. Any pre-existing `.db`
  files using the old names are dead storage and should be regenerated under SynPUF.

## External references

- Upstream OpenEnv framework: <https://github.com/meta-pytorch/OpenEnv> — authoritative
  for the Gym-style API (`reset`, `step`, `state`) and the `openenv init` / `openenv push`
  CLI.
- NotebookLM "The Future of AI and Industrial Innovation" (library id
  `the-future-of-ai-and-industria`) is registered in the `notebooklm` MCP server. Consult
  it before guessing on OpenEnv / vLLM MRV2 / GRPO / HF Spaces specifics.
