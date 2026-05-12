# I Built an AI Gymnasium for Catching $100B Medicare Fraud — Here's How FraudHunterEnv Works

> **A deep-dive into building a programmatic RLVR environment with a 7-layer reward system, adaptive case difficulty, and verifiable forensic agent traces**

---

*By Vansh Goel · OpenEnv Hackathon · May 2026*

---

Every year, the U.S. government loses an estimated **$100 billion** to Medicare and Medicaid fraud. Dead patients billed for surgeries. Shell companies funneling kickbacks. Duplicate claims filed minutes apart. Investigators know the patterns — but there are too many claims and too few humans.

This is exactly the kind of problem reinforcement learning was made for. Not pattern-matching on static datasets, but **multi-step reasoning under uncertainty** with verifiable ground truth.

I spent three months building **FraudHunterEnv**: an OpenEnv-compliant RL gymnasium that trains AI agents to investigate government fraud the same way a real qui tam investigator would — querying databases, linking shell companies, reading scanned claim forms, and building a proof chain before filing a False Claims Act complaint.

This article is a complete technical walkthrough. By the end, you'll understand how the environment works, why the reward design matters more than the model, and what a 91-point gap between a random policy and a scripted expert tells us about the signal quality of this environment.

---

## The Core Problem: LLMs Can Hallucinate Their Way to "Fraud Found"

Before I built anything, I had to answer a hard question: **why can't you just prompt GPT-4 to investigate fraud?**

You can. It'll confidently tell you about suspicious entities, flag unusual billing patterns, and recommend prosecution. The problem is you have no idea if any of it is real.

Standard LLM evaluation uses human judges or GPT-4-as-judge. For a forensic investigation environment, that creates a catastrophic failure mode: the model learns to *sound* convincing rather than to *be* correct. A hallucinated shell company with a plausible corporate name gets rewarded the same as a real one.

The solution is **RLVR — Reinforcement Learning with Verifiable Rewards**. Every reward signal comes from a programmatic grader with access to ground truth. You either found the right contradiction or you didn't. You either matched the NPI in the registry or you paid a penalty. There is no subjective evaluation.

---

## What Is FraudHunterEnv?

FraudHunterEnv is a Gym-style RL environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. Each episode is a self-contained fraud investigation case backed by a SQLite database following the CMS SynPUF (Medicare claims) schema.

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT (LLM Policy)                    │
│        <think>I see Acme Shell LLC...</think>            │
│              action: link_shell(...)                     │
└──────────────────────┬──────────────────────────────────┘
                       │ FraudHunterAction (Pydantic)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server                          │
│   /reset  /step  /state  /ws  /metrics  /leaderboard    │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────┐
│  Case Database   │    │     7-Layer RLVR Grader       │
│  (SQLite/SynPUF) │    │  Layer 1: Format Gate         │
│                  │    │  Layer 2: CoT Validity        │
│  carrier_claims  │    │  Layer 3: Step Decay          │
│  beneficiary_    │    │  Layer 4: Entity Extraction   │
│  corporate_reg   │    │  Layer 5: Shell Links         │
│  ground_truth    │    │  Layer 6: NPI Validation      │
│  loan_apps       │    │  Layer 7: Contradiction       │
└──────────────────┘    └──────────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────┐
                       │  FraudHunterObserv.  │
                       │  reward, done, info  │
                       │  tool_output         │
                       │  grader_feedback     │
                       │  proof_trace         │
                       └──────────────────────┘
```

The key design decision: **the agent never sees the ground truth table**. It must discover fraud the same way a real investigator would — by querying databases, reading evidence files, and building a logical proof chain.

---

## Architecture: Three Interlocking Systems

### 1. The Case Generator (`data_gen/`)

Each case is a fully synthetic universe seeded from a random integer. The same seed always produces the same case — reproducible, auditable, and impossible to memorize.

Four fraud typologies are implemented as real database generators (not just label swaps):

| Typology | What Gets Planted | Multiplier |
|---|---|---|
| **AKS Violation** | Shell-linked provider + inflated HCPCS billing | 1.6× |
| **Dead Patient Claim** | `BENE_DEATH_DT` before `CLM_FROM_DT` | 1.0× |
| **PPP Fraud** | Inflated `payroll_employees` in `loan_applications` | 1.8× |
| **Shell Chain** | 4+ layer `corporate_registry` hierarchy | 2.0× |

Tier gates difficulty: Tier 1 = clean claims + visible shell, Tier 5 = heavy OCR degradation + red herrings + multi-layer obfuscation. The `DifficultyManager` tracks a per-session 20-episode rolling reward and promotes/demotes automatically — **RLVE (RL with Verifiable Environments)**.

### 2. The 7-Layer RLVR Grader (`server/grader.py`)

This is the intellectual core of the project. Every action an agent takes passes through seven sequential evaluation layers, each contributing to or subtracting from the episode reward.

**Layer 1 — Format Gate**
Malformed JSON: `−10 pts + episode terminates`. No partial credit. Forces strict schema compliance from step one.

**Layer 2 — Chain-of-Thought Validity**
Every action must include a `<think>...</think>` block. Missing CoT: `−2 pts`. Grounded reasoning (mentions actual entities from the case): `+1 pt`. This layer also penalizes verbose padding (`−0.005 per word` over 50 words, phased out after step 20).

**Layer 3 — Step Decay**
`−0.1 pts` per step. Rewards efficiency. An agent that submits in 6 steps beats one that submits in 60 steps even with identical discoveries.

**Layer 4 — Entity Extraction**
Extracting a real entity from ground truth: `+10 pts`. Hallucinating one: `−50 pts`. The asymmetry is intentional — it makes hallucination a catastrophic failure, not a minor penalty.

**Layer 5 — Shell Link Confirmation**
Correctly linking a child entity to its true parent in the ownership chain: `+50 pts`. The grader verifies against `ground_truth WHERE kind='shell_link'`. Wrong link: `−20 pts`.

**Layer 6 — NPI Luhn Validation**
Provider extractions require a valid 10-digit NPI. The grader runs the [Luhn algorithm](https://www.cms.gov/Regulations-and-Guidance/Administrative-Simplification/NationalProvIdentStand) (`"80840" + npi[:-1]`) before any database lookup. Invalid NPI: `−20 pts`. Exact registry match: `+25 pts`.

**Layer 7 — Contradiction Confirmation**
The highest-value layer. Confirming a ground-truth contradiction (duplicate bill, dead patient claim, etc.) earns `+100 pts × typology_multiplier`. `foreign_affiliation` multiplies at `2.0×`. `ppp_fraud` at `1.8×`. Chaining a PDF document into the proof earns a further `1.5×` multiplier.

### 3. The Web Interface (`web/index.html`)

A single-file React-style dashboard that serves the live investigation UI. The "Qui Tam Fraud Investigator" interface connects to the FastAPI server over HTTP/SSE and displays:

- Real-time reward curve
- Proof trace log
- CoT trace panel
- Grader feedback stream
- NL action input (natural language → structured action via LLM)

---

## Live Demo: A Full Investigation Session

This is a real, unedited trace from running the grader against the demo case. Watch the reward accumulate as each layer fires.

### Step 0 — Case Brief (received on /reset)

```
You are a qui tam fraud investigator operating under the False Claims Act.
Case id: t1_deb21acf | Difficulty: Tier 1

A whistleblower alleges fraud involving shell companies, Medicare abuse,
government contracting irregularities, and/or PPP loan fraud.

Available tools:
  query_corporate(entity_name|entity_id)  → corporate registry + filings
  query_medicare(beneficiary_id|claim_id) → claims / beneficiary records
  sql_query(sql_statement)                → raw SELECT on the case database
  code_act(python_code)                   → sandboxed Python with conn, pd
  extract_entity(name, kind, npi_code?)   → flag an entity as fraudulent
  link_shell(child_entity, parent_entity) → assert UBO ownership
  claim_contradiction(evidence_a, b, kind)→ flag anomaly
  ocr_document(pdf_path)                  → OCR + base64 source document
  submit_case(case_summary, confidence)   → terminate and seek conviction

Budget: 60 steps. Format-gate: invalid JSON → -10 + episode ends.
```

### Step 1 — Map the Corporate Structure

```python
Action: sql_query
SQL: SELECT entity_id, entity_name, parent_entity_id, state
     FROM corporate_registry

┌────────┬──────────────────────────┬──────────────────┬───────┐
│ entity │ name                     │ parent_entity_id │ state │
├────────┼──────────────────────────┼──────────────────┼───────┤
│ E001   │ Starpoint Medical LLC    │ None             │ CA    │
│ E002   │ Starpoint Holdings Ltd   │ None             │ DE    │
│ E003   │ Acme Shell LLC           │ E002             │ NY    │
│ E_PROV │ John Doe MD              │ E001             │ CA    │
└────────┴──────────────────────────┴──────────────────┴───────┘

  reward : +1.90   feedback: sql_ok(4_rows)
```

*Acme Shell LLC has `parent_entity_id = E002`. This is the signal.*

### Step 2 — Drill Into the Shell

```python
Action: query_corporate("Acme Shell LLC")

entity_id=E003 name='Acme Shell LLC' tax_id=TX-003
parent=E002 ubo=U_001 registered=2020-01-01 state=NY npi=None

  reward : +0.90   feedback: cot_grounded(1_entities); corporate_registry_returned
```

### Step 3 — Confirm the Ownership Chain

```python
Action: link_shell(child="Acme Shell LLC", parent="Starpoint Holdings Ltd")

  reward : +51.90
  feedback: cot_grounded(2_entities);
            shell_link_confirmed:acme shell llc→starpoint holdings ltd
```

**51.9 points in one step.** The grader verified the child-parent pair against `ground_truth WHERE kind='shell_link'` and fired the `LINK_SHELL_REWARD = +50`.

### Step 4 — Flag the Duplicate Billing

```python
Action: claim_contradiction(
    evidence_a = "claim:C001",
    evidence_b = "claim:C002",
    contradiction_kind = "duplicate_bill"
)
# Think: C001 and C002 are identical — same date (2026-02-10),
#        same NPI 1234567890, same HCPCS 99214, same amount $480.

  reward : +99.90
  feedback: contradiction_confirmed:duplicate_bill(×1.0)
  proof_trace: ['contradiction:duplicate_bill:claim:c001↔claim:c002']
```

### Step 5 — Dead Patient Claim

```python
Action: claim_contradiction(
    evidence_a = "beneficiary:BENE_001",
    evidence_b = "claim:C003",
    contradiction_kind = "dead_patient_claim"
)
# Think: BENE_001 died 2025-11-03 but claim C003 dated 2026-03-01.
#        Billing 118 days after confirmed death.

  reward : +100.90
  feedback: cot_grounded(1_entities);
            contradiction_confirmed:dead_patient_claim(×1.0)
```

### Step 6 — Submit the Case

```python
Action: submit_case(
    case_summary = "Shell chain Acme Shell LLC → Starpoint Holdings Ltd confirmed.
                    Duplicate billing C001/C002. Post-mortem claim C003 against
                    deceased BENE_001. Recommend FCA prosecution.",
    confidence = 0.92,
    typologies = ["duplicate_bill", "dead_patient_claim", "foreign_affiliation"]
)
```

**Total cumulative reward after 6 steps: ~+255 pts.** Six steps. Two contradictions. One shell link. Zero hallucinations.

---

## The Reward Table

| Constant | Value | What Triggers It |
|---|---:|---|
| `FORMAT_GATE_PENALTY` | −10 | Invalid action schema |
| `COT_MISSING_PENALTY` | −2 | No `<think>` block |
| `STEP_DECAY` | −0.1 / step | Always |
| `HALLUCINATED_ENTITY_PENALTY` | −50 | Entity not in ground truth |
| `EXTRACT_ENTITY_REWARD` | +10 | Correct entity found |
| `LINK_SHELL_REWARD` | +50 | Correct ownership chain |
| `HALLUCINATED_LINK_PENALTY` | −20 | Wrong shell link |
| `NPI_EXACT_MATCH_BONUS` | +25 | Luhn-valid + registry match |
| `NPI_MISMATCH_PENALTY` | −20 | Luhn fail or wrong NPI |
| `CONTRADICTION_REWARD` | +100 | Ground-truth contradiction confirmed |
| `DOC_CLAIM_MATCH_BONUS` | +30 | OCR fields match claim record |
| `CODEACT_BONUS` | +5 | `code_act` reads a new evidence file |
| `CASE_WON_REWARD` | +1000 | All GT covered + typologies correct |
| `CASE_PARTIAL_REWARD` | +250 | Partial evidence, correct submit |

The `foreign_affiliation` typology multiplier of `2.0×` means a foreign shell company contradiction pays `+200 pts`. `ppp_fraud` pays `+180 pts`. These multipliers encode the *real-world prosecution priority* of different fraud schemes.

---

## Baseline vs Expert: 30-Episode Evaluation

After building the environment, I ran a 30-episode evaluation comparing two policies:

- **`random_policy`**: schema-valid random actions, no `<think>` traces, no SQL reasoning
- **`scripted_expert`**: hand-crafted SQL heuristics, full CoT, no answer-key access

```
=== Policy Evaluation (30 episodes, Tier 1) ===

  random_policy           reward = -73.05 ± 86.61   recall = 0.23
  scripted_expert_policy  reward = +18.13 ±  0.41   recall = 0.78
```

![Reward comparison chart](assets/baseline_vs_expert_episode_reward.png)

![Agentic recall chart](assets/baseline_vs_expert_agentic_recall.png)

The numbers tell a clear story:

1. **~91-point mean reward gap** — the expert consistently scores positive while random collapses
2. **3.4× recall improvement** — the expert recovers 78% of ground-truth evidence vs 23% for random
3. **Near-zero expert variance (±0.41)** — the grader is deterministic; good reasoning consistently wins
4. **High random variance (±86.61)** — random occasionally stumbles into a contradiction, but it's noise

This is the signal an RL training loop needs. Not a human judge. Not another LLM. A deterministic programmatic score that rewards *actual investigative work*.

---

## What Makes This Technically Different

### The CodeAct Sandbox
Agents can submit arbitrary Python code through `code_act`. The sandbox provides a read-only SQLite facade (write operations raise `PermissionError`), path-confined filesystem helpers (`open`, `listdir` scoped to the case directory), and a 5-second timeout enforced via `sys.settrace`. An agent that discovers `intercepted_comms/email_00.txt` earns a `+5 pt` bonus — rewarding proactive evidence hunting.

### Luhn NPI Validation
Every provider extraction is validated against the CMS NPI Luhn algorithm before any database lookup. This forces the model to submit valid NPIs, not plausible-sounding 10-digit strings. The asymmetric reward (`+25` for match, `−20` for mismatch) creates a strong incentive for precision.

### Adaptive Difficulty (RLVE)
The `DifficultyManager` tracks rolling episode performance per session. Too many wins → promoted to Tier 2 (multi-layered shells + OCR-required evidence). Too many losses → demoted. The agent never trains on a fixed curriculum; it always faces appropriate resistance. This mirrors [RLVE (RL with Verifiable Environments)](https://arxiv.org/abs/2504.16084).

### Proof Chain Accumulation
The grader builds a `proof_trace` list across the episode. The `PROOF_CHAIN_MULTIPLIER = 1.5×` fires at submission when the chain is complete. An agent that submits without a chain gets no multiplier. An agent that builds one step-by-step earns 50% more on the final reward. This rewards structured *reasoning under evidence*, not just lucky guesses.

---

## The Evaluation Pitfall I Almost Missed

Early in development, the scripted expert policy contained this query:

```python
"SELECT payload_json FROM ground_truth WHERE kind='entity' LIMIT 1"
```

The policy would read the answer key, extract the entity name, and immediately `extract_entity`. 100% recall. Suspicious zero variance. Numbers that looked too good.

This is **answer-key cheating** — a subtle form of data leakage where the evaluation policy reads the ground truth directly. After discovering this via git bisect, I replaced it with a legitimate heuristic: query `corporate_registry WHERE parent_entity_id IS NOT NULL`. The policy now earns its score by reasoning about corporate structure, not by peeking at the answer.

The lesson: **evaluation integrity is as important as model performance**. A fraudulent eval is worse than no eval.

---

## Building Your Own RL Environment: 5 Lessons

**1. Design reward asymmetry deliberately.** Hallucination penalty (−50) is 5× the extraction reward (+10). This isn't arbitrary — it encodes the real-world cost of false accusations in fraud investigation.

**2. Verifiable rewards beat human judges.** The 30-episode eval took 73 seconds with zero human involvement. Every reward came from a function with deterministic output.

**3. Ground truth must be hidden from the policy.** Build a separate `ground_truth` table and ensure no query path exposes it to the agent. Test with a "cheating oracle" and make sure it outperforms everything else — if it doesn't, your GT is wrong.

**4. Adaptive difficulty prevents reward hacking.** A fixed Tier 1 environment gets gamed. The 5-tier curriculum forces generalization.

**5. The CoT layer is not optional.** The `<think>` requirement isn't just a nice-to-have — it's what makes the reward signal interpretable during training. You need to see *why* the model earned the reward to improve it.

---

## What's Next: Training a Real Agent

The environment is training-ready. The next step is plugging in a real LLM policy and running GRPO (Group Relative Policy Optimization) to fine-tune it on the reward signal.

```python
# Training loop (pseudocode)
env = FraudHunterEnv(base_url="http://localhost:8000")
policy = LlamaPolicy("meta-llama/Llama-3.2-11B-Vision-Instruct")

for episode in range(N_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS):
        action = policy.sample(obs)          # LLM generates action
        obs, reward, done, info = env.step(action)
        policy.record(action, reward)        # GRPO buffer
        if done:
            break
    policy.update()                          # GRPO gradient step
```

The `grpo_train.ipynb` notebook in `training/` contains the full GRPO loop with Llama 3.2 11B Vision-Instruct — a multimodal model that can read the scanned claim PDFs the same way a human investigator would.

---

## Technical Stack

| Component | Technology |
|---|---|
| Environment API | FastAPI + WebSocket (OpenEnv contract) |
| Data Schema | CMS SynPUF (Medicare claims) |
| Database | SQLite per case |
| Case Generation | Python + Pillow (PDF scan synthesis) |
| OCR | Tesseract / pdfplumber |
| Online RL | Contextual bandit (softmax policy gradient) |
| Containerization | Docker (non-root, `appuser:1001`) |
| Testing | pytest, 119 tests, 0 failures |
| Training | GRPO via Llama 3.2 11B VLM (notebook) |

---

## Reproducibility

```bash
# Clone and install
git clone <repo>
cd fraud_hunter_env
uv sync

# Start the server
uvicorn fraud_hunter_env.server.app:app --port 8000

# Run the evaluation (30 episodes, writes assets/*.png + eval_results.json)
python eval.py --episodes 30 --tier 1

# Run all tests
python -m pytest tests/ -q
# → 119 passed in 73s
```

The environment runs entirely locally with no external API calls required. No Hugging Face token, no OpenAI key, no cloud dependency.

---

## Closing Thoughts

The $100B Medicare fraud problem isn't going to be solved by a chatbot that confidently names suspicious entities. It's going to be solved by agents that can reason step-by-step through structured evidence, validate NPIs against real registries, link shell companies across jurisdictions, and build a proof chain that holds up in court.

FraudHunterEnv is my attempt to build the training ground for that agent.

The 91-point gap between random and expert on a programmatic grader is not a benchmark number to brag about. It's a *learning signal*. It tells an RL training loop exactly what good forensic reasoning looks like, and it does so 73 times per second.

That's what RLVR makes possible.

---

*The full environment, all 119 tests, and the GRPO training notebook are available in the project repository.*

*Questions? Find me at the OpenEnv Hackathon Discord or open an issue on GitHub.*

---

## Visual Guide for Publication

**Recommended toolchain for publishing this on Medium:**

1. **Header image** — Use [Canva](https://canva.com). Search template: "FBI Investigation Dark Theme" or "Cybersecurity Abstract". Overlay text: *"FraudHunterEnv — Teaching AI to Catch Medicare Fraud"*. Color palette: deep navy (#111827), amber (#F59E0B), white.

2. **Architecture diagram** — Use [Napkin AI](https://napkin.ai) — paste the ASCII diagram above and it auto-renders to a clean flowchart SVG. Export as PNG for Medium.

3. **Reward table** — Screenshot this article's markdown table directly in VS Code with a dark theme (Dracula or One Dark Pro). The table renders beautifully.

4. **Terminal screenshots** — Open the `fraud_hunter_env` project in VS Code. Run the grader demo (`python -m pytest tests/test_grader.py -v -s`). Screenshot the terminal output with your OS dark terminal (Windows Terminal with Dracula theme).

5. **Eval charts** — Already generated at `assets/baseline_vs_expert_episode_reward.png` and `assets/baseline_vs_expert_agentic_recall.png`. Upload directly to Medium.

---

**SEO metadata (use as Medium subtitle/tags):**

- Primary keyword: *reinforcement learning fraud detection*
- Secondary: *RLVR Medicare AI*, *AI fraud investigation environment*, *OpenEnv hackathon*
- Tags: `Artificial Intelligence`, `Machine Learning`, `Healthcare`, `Python`, `Reinforcement Learning`
- Estimated read time: 12 minutes
