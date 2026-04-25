"""
Baseline-vs-scripted-expert evaluation harness.

Runs N episodes for each of two policies against the FraudHunterEnvironment
and writes:

  assets/baseline_vs_expert_reward.png     — mean episode reward + 95% CI
  assets/baseline_vs_expert_recall.png     — mean agentic recall
  assets/baseline_vs_expert_cot.png        — mean CoT validity
  assets/eval_results.json                 — raw per-episode metrics

This is the substrate the README plots are built from. Once a real RL-trained
policy exists (see ``training/grpo_train.py``), drop it in as a third policy
in ``POLICIES`` below and re-run to produce the trained-vs-baseline plot.

Usage:
    python eval.py --episodes 30 --tier 1
    python eval.py --episodes 50            # adaptive (RLVE) tier per session
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from fraud_hunter_env.models import FraudHunterAction
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment


# ── Policies ─────────────────────────────────────────────────────────────────
# A "policy" is a callable: (observation, env) -> dict action payload.
# Returning None ends the episode.

def random_policy(obs, env, step: int) -> dict | None:
    """Untrained baseline: sample a random schema-valid action with random fields.

    Roughly mimics what a zero-shot LLM might emit if it had not been trained
    on the format. Mostly produces format-gate failures and shallow SQL.
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


def scripted_expert_policy(obs, env, step: int) -> dict | None:
    """Hand-crafted upper-bound policy that uses ground-truth knowledge of the
    case schema. This represents what a *well-trained* agent could plausibly
    achieve and is the moral equivalent of a "trained-policy" stand-in until
    we ship the real RL checkpoint.

    Strategy: enumerate registry → query a beneficiary → run an aggregate SQL →
    extract one entity → submit.
    """
    if env.case is None:
        return {"kind": "submit_case", "case_summary": "no case", "confidence": 0.0}

    conn = env.case.conn

    if step == 0:
        row = conn.execute("SELECT entity_name FROM corporate_registry LIMIT 1").fetchone()
        if row:
            return {"kind": "query_corporate", "entity_name": row[0]}

    if step == 1:
        try:
            row = conn.execute(
                "SELECT DESYNPUF_ID FROM beneficiary_summary LIMIT 1"
            ).fetchone()
            if row:
                return {"kind": "query_medicare", "beneficiary_id": row[0]}
        except Exception:
            pass

    if step == 2:
        # SQL exploration over a typology-relevant table.
        return {
            "kind": "sql_query",
            "sql_statement": (
                "SELECT entity_name, COUNT(*) FROM corporate_registry "
                "GROUP BY entity_name LIMIT 10"
            ),
        }

    if step == 3:
        row = conn.execute("SELECT entity_name FROM corporate_registry LIMIT 1").fetchone()
        if row:
            return {
                "kind": "extract_entity",
                "extracted_name": row[0],
                "extracted_kind": "corporation",
            }

    if step == 4:
        return {
            "kind": "submit_case",
            "case_summary": "Scripted expert traversal complete.",
            "confidence": 0.85,
        }

    return None  # episode done


PolicyFn = Callable[[Any, FraudHunterEnvironment, int], dict | None]

POLICIES: dict[str, PolicyFn] = {
    "random_baseline": random_policy,
    "scripted_expert": scripted_expert_policy,
}


# ── Eval loop ────────────────────────────────────────────────────────────────

def run_episode(env: FraudHunterEnvironment, policy: PolicyFn) -> dict:
    """Run one episode and return its terminal metrics dict."""
    env.reset()
    last_obs = None
    episode_reward = 0.0
    for step in range(64):
        # Best-effort: build an obs view for the policy from the env's state.
        # We pass the env itself so policies can introspect the case DB.
        try:
            payload = policy(last_obs, env, step)
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
            obs = env.step(action)
        except Exception:
            break
        episode_reward += float(obs.reward or 0.0)
        last_obs = obs
        if obs.done:
            break

    metrics = env._build_metrics()  # noqa: SLF001 — eval harness, intentional
    metrics["episode_reward"] = round(episode_reward, 2)
    return metrics


def run_policy(name: str, policy: PolicyFn, n_episodes: int, tier: int | None) -> list[dict]:
    print(f"\n=== Policy: {name} ({n_episodes} episodes) ===")
    results = []
    env = FraudHunterEnvironment()
    for i in range(n_episodes):
        # Force a tier by pinning the per-session profile when --tier is set.
        if tier is not None:
            profile = env._diff_mgr.get_or_create(env._session_id)  # noqa: SLF001
            profile.current_tier = tier
        m = run_episode(env, policy)
        results.append(m)
        if (i + 1) % max(1, n_episodes // 10) == 0:
            print(f"  ep {i+1:>3}/{n_episodes}  reward={m['episode_reward']:>8.2f}  "
                  f"recall={m['agentic_recall']:.2f}  cot={m['cot_validity_score']:.2f}")
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
    for key, title in metrics:
        names = list(all_results.keys())
        means = [_summary([r[key] for r in all_results[n]])[0] for n in names]
        stds = [_summary([r[key] for r in all_results[n]])[1] for n in names]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, means, yerr=stds, capsize=8, color=["#cc4444", "#44aa66"])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tier", type=int, default=None,
                    help="Force a difficulty tier (1-5). Default: RLVE adaptive.")
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    all_results: dict[str, list[dict]] = {}
    for name, policy in POLICIES.items():
        all_results[name] = run_policy(name, policy, args.episodes, args.tier)

    # Summary
    print("\n=== Summary ===")
    for name, results in all_results.items():
        rewards = [r["episode_reward"] for r in results]
        recalls = [r["agentic_recall"] for r in results]
        m_r, s_r = _summary(rewards)
        m_a, s_a = _summary(recalls)
        print(f"  {name:>20}  reward = {m_r:>8.2f} ± {s_r:>5.2f}   "
              f"recall = {m_a:.2f} ± {s_a:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_results.json").write_text(json.dumps(all_results, indent=2))
    write_plots(all_results, args.out)


if __name__ == "__main__":
    main()
