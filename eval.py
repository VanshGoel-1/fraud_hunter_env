"""
Baseline-vs-scripted-expert evaluation harness — HTTP/WS client edition.

Drives a running Fraud Hunter Env server through the OpenEnv WebSocket
contract (no in-process imports of FraudHunterEnvironment). This is the
canonical client-server demo: every action the policy emits crosses the
network, every reward comes from the server's grader, every metric is what
the dashboard sees.

Outputs (regenerated per run):
  assets/baseline_vs_expert_episode_reward.png       — bar chart, mean ± std
  assets/baseline_vs_expert_agentic_recall.png       — bar chart, mean ± std
  assets/baseline_vs_expert_cot_validity_score.png   — bar chart, mean ± std
  assets/eval_results.json                           — per-episode metrics

Usage (server must be running first):
    uv run server/app.py &
    python eval.py --episodes 30 --tier 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import requests

from fraud_hunter_env.client import FraudHunterEnv
from fraud_hunter_env.models import FraudHunterAction
from fraud_hunter_env.replay import verify_replay_chain
from fraud_hunter_env.server.http_contract import ensure_server_up


EVAL_SEED_RANGE = (8001, 10000)


# ── Policy state ─────────────────────────────────────────────────────────────
# Policies are stateful within an episode (they remember what they discovered
# from prior tool_outputs). State is a plain dict the harness threads through.

PolicyFn = Callable[[Any, dict, int], dict | None]


# ── Random baseline policy ───────────────────────────────────────────────────

def random_policy(obs, state: dict, step: int) -> dict | None:
    """Untrained baseline: random schema-valid actions, no <think>, no
    grounding. Mostly produces format-gate failures and shallow SQL.
    Mirrors what a zero-shot LLM would emit if untrained on the format.
    """
    pool = [
        {"kind": "query_corporate", "entity_name": f"Random Corp {random.randint(0,9)}"},
        {"kind": "query_medicare", "beneficiary_id": f"B{random.randint(0,99):03d}"},
        {"kind": "sql_query", "sql_statement": "SELECT * FROM corporate_registry LIMIT 5"},
        {"kind": "extract_entity",
         "extracted_name": f"Random Corp {random.randint(0,9)}",
         "extracted_kind": "corporation"},
        {"kind": "submit_case",
         "case_summary": "Random submission.",
         "confidence": 0.5},
    ]
    return random.choice(pool)


# ── Scripted-expert policy ────────────────────────────────────────────────────
# An "upper bound" hand-crafted agent. Uses only legitimate SQL queries against
# the case database — no ground_truth peeking. Entity extraction is heuristic:
# finds shell companies via parent_entity_id links in corporate_registry.

_EXPERT_QUERIES: list[tuple[str, str]] = [
    ("Search for shell-company chains sharing a UBO.",
     "SELECT entity_name, ubo_id FROM corporate_registry "
     "GROUP BY ubo_id HAVING COUNT(*) > 1 LIMIT 5"),
    ("Find beneficiaries with death dates (dead-patient claims).",
     "SELECT DESYNPUF_ID, BENE_DEATH_DT FROM beneficiary_summary "
     "WHERE BENE_DEATH_DT IS NOT NULL LIMIT 5"),
    ("Look for duplicate carrier claims.",
     "SELECT CLM_ID, COUNT(*) FROM carrier_claims "
     "GROUP BY CLM_ID HAVING COUNT(*) > 1 LIMIT 5"),
    ("Identify high-volume prescribers (potential AKS).",
     "SELECT PRVDR_NPI, COUNT(*) c FROM prescription_drug_events "
     "GROUP BY PRVDR_NPI ORDER BY c DESC LIMIT 5"),
    ("Pull referral payments (AKS smoking-gun ledger).",
     "SELECT payer_npi, payee_npi, amount FROM referral_payments LIMIT 5"),
    ("Check general ledger for anomalous transactions.",
     "SELECT memo, amount FROM general_ledger ORDER BY amount DESC LIMIT 5"),
    ("Index evidence documents (PDFs).",
     "SELECT doc_id, claim_id FROM evidence_documents LIMIT 5"),
    ("Find shell entities via parent_entity_id links (legitimate heuristic).",
     "SELECT entity_name FROM corporate_registry "
     "WHERE parent_entity_id IS NOT NULL LIMIT 1"),
]


def _parse_entity_name(tool_output: str | None) -> str | None:
    """Extract an entity name from a sql_query tool_output row."""
    if not tool_output:
        return None
    for line in tool_output.splitlines():
        line = line.strip().strip("|").strip()
        if line and not line.startswith("-") and not line.lower().startswith("entity_name"):
            return line
    return None


def scripted_expert_policy(obs, state: dict, step: int) -> dict | None:
    """Hand-crafted upper bound: 8 SQL queries hitting all required source
    tables, then extract a shell entity heuristically + submit. Every action
    wraps a <think> block so the CoT-validity layer rewards it.
    Network-only — no in-process DB peeking, no ground_truth access.
    """
    if step < len(_EXPERT_QUERIES):
        thought, sql = _EXPERT_QUERIES[step]
        return {
            "kind": "sql_query",
            "sql_statement": sql,
            "think_trace": f"<think>{thought}</think>",
        }

    if step == len(_EXPERT_QUERIES):
        # Last obs was the shell-entity query; parse the name heuristically.
        if obs is not None:
            tool_output = getattr(obs, "tool_output", None)
            name = _parse_entity_name(tool_output)
            if name:
                state["shell_name"] = name
        target = state.get("shell_name")
        if target:
            return {
                "kind": "extract_entity",
                "extracted_name": target,
                "extracted_kind": "corporation",
                "think_trace": (
                    f"<think>corporate_registry shows {target} has a parent entity — "
                    f"shell indicator. Extracting as a fraudulent corporation.</think>"
                ),
            }

    if step == len(_EXPERT_QUERIES) + 1:
        return {
            "kind": "submit_case",
            "case_summary": "Scripted expert traversal complete.",
            "confidence": 0.85,
            "think_trace": "<think>All required tables queried; submitting.</think>",
        }

    return None


POLICIES: dict[str, PolicyFn] = {
    "random_policy": random_policy,
    "scripted_expert_policy": scripted_expert_policy,
}


# ── Server health pre-check ──────────────────────────────────────────────────
# Health probe lives in server/http_contract.py — eval.py and inference.py
# both call it directly so the wire-format check is single-source.


def _configure_eval_seed_range(base_url: str) -> None:
    try:
        response = requests.post(
            f"{base_url}/fraud_hunter/seed_range",
            json={"seed_min": EVAL_SEED_RANGE[0], "seed_max": EVAL_SEED_RANGE[1]},
            timeout=3,
        )
        response.raise_for_status()
    except Exception as exc:
        sys.stderr.write(
            f"\n[FATAL] could not configure evaluation seed range {EVAL_SEED_RANGE}: {exc}\n"
        )
        sys.exit(2)


# ── Episode loop ─────────────────────────────────────────────────────────────

def run_episode(client: FraudHunterEnv, policy: PolicyFn, max_steps: int = 64) -> dict:
    """Run one episode end-to-end through the HTTP/WS contract.

    Returns the terminal `obs.info` dict (which the env populates with
    agentic_recall, cot_validity_score, format_error_count, episode_reward,
    proof_chain_length, difficulty_tier, hits, case_id).
    """
    result = client.reset()
    obs = result.observation
    state: dict = {}
    episode_reward = 0.0
    last_info: dict[str, Any] = {}
    info_history: list[dict] = []
    if obs and obs.info:
        info_history.append(dict(obs.info))

    for step in range(max_steps):
        try:
            payload = policy(obs, state, step)
        except Exception:
            break
        if payload is None:
            break
        try:
            action = FraudHunterAction.model_validate(payload)
        except Exception:
            episode_reward -= 10.0
            break
        try:
            result = client.step(action)
        except Exception:
            break
        obs = result.observation
        episode_reward += float(result.reward or 0.0)
        if obs and obs.info:
            last_info = dict(obs.info)
            info_history.append(dict(obs.info))
        if result.done:
            break

    last_info.setdefault("episode_reward", round(episode_reward, 2))
    # Trust the server's terminal metrics if present, else fall back to our tally.
    if "episode_reward" not in last_info:
        last_info["episode_reward"] = round(episode_reward, 2)

    # Audit the SHA-256 replay chain emitted by the env per step. A break
    # indicates either tampering or out-of-order info records — surface but
    # don't abort the eval (a single bad chain shouldn't void the whole run).
    chain_ok, chain_reason = verify_replay_chain(info_history)
    last_info["replay_chain_ok"] = chain_ok
    last_info["replay_chain_steps"] = len(info_history)
    if not chain_ok:
        last_info["replay_chain_reason"] = chain_reason
        sys.stderr.write(f"[replay] chain verification failed: {chain_reason}\n")
    return last_info


def run_policy(
    base_url: str,
    name: str,
    policy: PolicyFn,
    n_episodes: int,
    tier: int | None,
) -> list[dict]:
    print(f"\n=== Policy: {name} ({n_episodes} episodes via {base_url}) ===")
    results = []
    # One client (one WS connection) per policy; reset() inside cycles episodes.
    # EnvClient is async by default; use .sync() for the synchronous wrapper.
    with FraudHunterEnv(base_url=base_url).sync() as client:
        for i in range(n_episodes):
            m = run_episode(client, policy)
            # Tier override is informational here; the server's RLVE manager picks
            # tiers per-session. To force a fixed tier in v1 of this harness, you
            # would need an admin endpoint — out of scope for the 5-hour build.
            results.append(m)
            if (i + 1) % max(1, n_episodes // 10) == 0 or (i + 1) == n_episodes:
                rec = m.get("agentic_recall", 0.0)
                cot = m.get("cot_validity_score", 0.0)
                rew = m.get("episode_reward", 0.0)
                print(f"  ep {i+1:>3}/{n_episodes}  reward={rew:>8.2f}  "
                      f"recall={rec:.2f}  cot={cot:.2f}")
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def _summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return m, s


def write_plots(all_results: dict[str, list[dict]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        print("Install with:  pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("episode_reward", "Mean Episode Reward"),
        ("agentic_recall", "Mean Agentic Recall"),
        ("cot_validity_score", "Mean CoT Validity"),
    ]
    palette = ["#cc4444", "#44aa66", "#4477cc", "#aa66dd"]
    for key, title in metrics:
        names = list(all_results.keys())
        means = [_summary([r.get(key, 0.0) for r in all_results[n]])[0] for n in names]
        stds = [_summary([r.get(key, 0.0) for r in all_results[n]])[1] for n in names]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, means, yerr=stds, capsize=8,
                      color=palette[: len(names)])
        ax.set_ylabel(title)
        ax.set_title(f"{title} — {len(all_results[names[0]])} episodes per policy")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        out = out_dir / f"baseline_vs_expert_{key}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


# ── Per-episode metric flattening for eval_results.json ──────────────────────

def _flatten_for_dashboard(results: list[dict]) -> dict[str, list[float]]:
    """The dashboard reads `data[policy][metric]` as a list. Pivot the list of
    per-episode dicts into that shape."""
    out: dict[str, list[float]] = {}
    keys = {
        "episode_reward", "agentic_recall", "cot_validity_score",
        "format_error_count", "proof_chain_length", "difficulty_tier",
        "hallucination_count",
    }
    for k in keys:
        out[k] = [float(r.get(k, 0.0)) for r in results]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tier", type=int, default=None,
                    help="(Informational) Force a difficulty tier (1-5). "
                         "Server's RLVE manager controls tier per session.")
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="Fraud Hunter Env server URL")
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        ensure_server_up(args.base_url)
    except Exception as exc:
        sys.stderr.write(
            f"\n[FATAL] cannot reach Fraud Hunter Env server at {args.base_url}: {exc}\n"
            f"Start it first:  uv run server/app.py\n\n"
        )
        sys.exit(2)

    # Warn if the case bank is too small to cover the eval seed range.
    from fraud_hunter_env import config as _cfg
    _bank = _cfg.case_bank_dir()
    if _bank.is_dir():
        _db_count = sum(1 for p in _bank.rglob("medicare_records.db"))
        _seed_span = EVAL_SEED_RANGE[1] - EVAL_SEED_RANGE[0]
        if _db_count < _seed_span:
            sys.stderr.write(
                f"[WARNING] case bank has {_db_count} cases but eval seed range spans "
                f"{_seed_span} seeds ({EVAL_SEED_RANGE}). "
                f"On-the-fly generation will cover the gap.\n"
            )

    _configure_eval_seed_range(args.base_url)

    all_results: dict[str, list[dict]] = {}
    for name, policy in POLICIES.items():
        all_results[name] = run_policy(
            args.base_url, name, policy, args.episodes, args.tier,
        )

    # Console summary
    print("\n=== Summary ===")
    for name, results in all_results.items():
        rewards = [r.get("episode_reward", 0.0) for r in results]
        recalls = [r.get("agentic_recall", 0.0) for r in results]
        m_r, s_r = _summary(rewards)
        m_a, s_a = _summary(recalls)
        print(f"  {name:>26}  reward = {m_r:>8.2f} ± {s_r:>5.2f}   "
              f"recall = {m_a:.2f} ± {s_a:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    # Pivoted shape that web/index.html expects.
    flat = {name: _flatten_for_dashboard(results)
            for name, results in all_results.items()}
    (args.out / "eval_results.json").write_text(json.dumps(flat, indent=2))
    print(f"wrote {args.out / 'eval_results.json'}")

    write_plots(all_results, args.out)


if __name__ == "__main__":
    main()
