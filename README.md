# 🔍 Government Fraud Hunter AI

> **An RLVR-based Reinforcement Learning Environment for Qui Tam False Claims Act Investigations**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.dev)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🏆 Competition Differentiators

| Feature | Basic Submission | **This Submission** |
|---------|:---:|:---:|
| Reward Model | LLM-as-a-judge | **RLVR: 7-layer programmatic grader** |
| Difficulty | Static cases | **RLVE: Adaptive 5-tier curriculum** |
| Agent Arch | JSON actions only | **CoT + CodeAct + MCP-style SQL** |
| NPI Validation | Partial credit | **Zero-tolerance Luhn checksum** |
| Fraud Types | 2 types | **12 FCA typologies** |
| Data Gen | Hardcoded | **Benford + Pareto distributions** |
| Training | Basic PPO | **DAPO loss + clip-higher + zero KL** |
| Eval Metrics | Avg reward | **CoT-Pass@K + Agentic Recall** |
| Dashboard | None | **Live SSE reward curves + trace viewer** |

## 📐 Architecture

```
┌─────────────────────────────────────────────────┐
│  Agent (LLM with LoRA)                          │
│  <think>CoT reasoning</think>                   │
│  {"kind":"query_corporate","entity_name":"..."} │
└────────────────┬────────────────────────────────┘
                 │ WebSocket (OpenEnv protocol)
                 ▼
┌─────────────────────────────────────────────────┐
│  FraudHunterEnvironment                         │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ RLVR      │  │ RLVE     │  │ CodeAct      │  │
│  │ Grader    │  │Difficulty│  │ Sandbox      │  │
│  │ (7 layers)│  │ Manager  │  │ (Python/SQL) │  │
│  └───────────┘  └──────────┘  └──────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │ SQLite Case Bank (5 tiers × N cases)      │  │
│  │ Benford amounts │ Pareto tails │ Red      │  │
│  │ Valid NPIs      │ 12 typologies│ Herrings │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 🧪 RLVR: 7-Layer Hierarchical Grader

1. **Format Gate** — Invalid JSON → `-10.0`, episode terminates
2. **CoT Enforcement** — Missing `<think>` → `-2.0` soft penalty
3. **CoT Grounding** — Entity names verified against DB → `+1.0` per grounded entity
4. **Length Schedule** — Excess CoT tokens penalized early (phases out at step 20)
5. **Duplicate Detection** — Repeat queries → `-5.0`
6. **NPI Strict Validation** — Provider NPI must match ground truth exactly
7. **Typology Matrix** — Per-typology difficulty multipliers (AKS ×1.6, PPP ×1.8)

**Bonus**: Process-Based Scoring — complete entity→link→contradiction proof chain → `×1.5` on terminal reward

## 🎯 RLVE: Adaptive Verifiable Environments

The environment dynamically scales difficulty based on the agent's rolling 20-episode average reward:

| Tier | Entities | Typologies | Shell Depth | Red Herrings |
|:----:|:--------:|:----------:|:-----------:|:------------:|
| 1 | 2 | 2 | 0 | No |
| 2 | 4 | 3 | 1 | No |
| 3 | 6 | 4 | 2 | Yes |
| 4 | 8 | 6 | 3 | Yes |
| 5 | 10 | 7 | 4 | Yes |

## 🔧 Quick Start

```bash
# 1. Install
uv sync

# 2. Generate tiered case bank
python data_gen/case_compiler.py --count 5

# 3. Run tests (25 tests covering all RLVR layers)
PYTHONPATH=. pytest tests/ -v

# 4. Start environment server
uv run --project . server

# 5. Open dashboard
# → http://localhost:8000/web
```

## 🤖 Agent Action Space

| Action | Description | Reward |
|--------|-------------|--------|
| `query_corporate` | Lookup corporate registry | (info only) |
| `query_medicare` | Lookup claims/beneficiaries | (info only) |
| `sql_query` | Raw SELECT on case DB (MCP-style) | +0.5/row |
| `code_act` | Sandboxed Python execution | +5.0/correct row |
| `extract_entity` | Flag entity as fraudulent | +10.0 (with NPI: +25.0) |
| `link_shell` | Assert UBO/shell relationship | +50.0 |
| `claim_contradiction` | Flag billing anomaly | +100.0 × typology multiplier |
| `submit_case` | Terminal: seek conviction | +1000.0 (won) / +250.0 (partial) |

## 📊 Fraud Typologies

**Healthcare**: Dead patient claims, duplicate billing, upcoding, unbundling, phantom beneficiaries, AKS kickbacks, off-label marketing

**Government Contracting**: Double-billing, cost/pricing fraud, product substitution

**Pandemic/PPP**: Misrepresented employee counts, undisclosed foreign affiliations

## 🏋️ Training (GRPO + DAPO)

```python
# Key DAPO configuration
GRPOConfig(
    loss_type="dapo",           # Normalize by active tokens
    epsilon=0.2,                # Standard PPO clip
    epsilon_high=0.25,          # Clip-higher prevents entropy collapse
    beta=0.0,                   # Zero KL penalty
    num_generations=8,          # For CoT-Pass@K evaluation
)
```

Advanced features:
- **Gibberish Detection**: Reject completions with >10% rare tokens
- **Rejection Sampling**: Top-30% episodes recycled as SFT data
- **CoT-Pass@K**: Evaluates reasoning validity across K completions
- **Agentic Recall**: Measures table exploration efficiency

## 🤗 Hugging Face Datasets

The synthetic case banks (`.db` files) and raw tabular datasets (`.csv` files) are configured to be tracked and pushed to Hugging Face. This allows for seamless remote model training, evaluation, and testing.

To push your locally generated datasets to the Hugging Face Hub:
```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Upload the data directory to your dataset repository
huggingface-cli upload <your-username>/fraud-hunter-cases ./data . --repo-type dataset
```

To train agents on the cloud, you can pull this dataset down via the `datasets` library or the CLI.

## 📂 Project Structure

```text
fraud_hunter_env/
├── data_gen
│   ├── build_case_bank.py
│   ├── case_compiler.py         # Tiered case generator (Benford/Pareto)
│   ├── pdf_evidence.py
│   └── __init__.py
├── server
│   ├── app.py                   # FastAPI + SSE + dashboard
│   ├── data_loader.py           # Tiered case bank loader
│   ├── difficulty.py            # RLVE adaptive difficulty manager
│   ├── fraud_hunter_env_environment.py  # Core environment
│   ├── grader.py                # 7-layer RLVR grader
│   ├── requirements.txt
│   ├── sandbox.py               # CodeAct Python/SQL sandbox
│   └── __init__.py
├── tests
│   ├── test_environment.py
│   ├── test_grader.py           # RLVR grader tests (25 total)
│   ├── test_models.py           # Schema validation tests
│   └── __init__.py
├── training
│   ├── grpo_train.ipynb
│   ├── grpo_train.py            # DAPO + clip-higher + rejection sampling
│   └── __init__.py
├── web
│   └── index.html               # RLVR monitoring dashboard
├── .gitignore
├── client.py                    # WebSocket client
├── Dockerfile
├── inference.py
├── models.py                    # Pydantic schemas + RLVR reward constants
├── openenv.yaml
├── pyproject.toml
├── README.md
├── uv.lock
└── __init__.py
```
