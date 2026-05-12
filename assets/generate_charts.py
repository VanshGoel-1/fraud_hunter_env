"""Generate all article charts from real eval_results.json data."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0F172A"   # deep navy
PANEL   = "#1E293B"
GRID    = "#334155"
AMBER   = "#F59E0B"
CYAN    = "#22D3EE"
RED     = "#F87171"
GREEN   = "#4ADE80"
WHITE   = "#F8FAFC"
MUTED   = "#94A3B8"

ASSETS = Path(__file__).parent
data   = json.loads((ASSETS / "eval_results.json").read_text())

rnd  = data["random_policy"]
exp  = data["scripted_expert_policy"]
eps  = list(range(1, len(rnd["episode_reward"]) + 1))

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    return fig, ax

def dark_fig2(w=14, h=5):
    fig, axes = plt.subplots(1, 2, figsize=(w, h), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    return fig, axes

# ─────────────────────────────────────────────────────────────────────────────
# 1. Episode reward — scatter + mean band
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = dark_fig(11, 5.5)

r_rewards = rnd["episode_reward"]
e_rewards = exp["episode_reward"]
r_mean = np.mean(r_rewards)
e_mean = np.mean(e_rewards)

ax.scatter(eps, r_rewards, color=RED,   s=40, alpha=0.75, zorder=3, label="random_policy")
ax.scatter(eps, e_rewards, color=GREEN, s=40, alpha=0.85, zorder=3, label="scripted_expert")
ax.axhline(r_mean, color=RED,   linewidth=1.5, linestyle="--", alpha=0.6)
ax.axhline(e_mean, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.6)
ax.axhline(0,      color=MUTED, linewidth=0.8, linestyle=":",  alpha=0.5)

ax.fill_between(eps, [r_mean]*len(eps), r_rewards, alpha=0.08, color=RED)
ax.fill_between(eps, [e_mean]*len(eps), e_rewards, alpha=0.08, color=GREEN)

ax.annotate(f"mean = {r_mean:+.1f}", xy=(len(eps)*0.72, r_mean+6),
            color=RED, fontsize=9, fontweight="bold")
ax.annotate(f"mean = {e_mean:+.1f}", xy=(len(eps)*0.72, e_mean-10),
            color=GREEN, fontsize=9, fontweight="bold")
ax.annotate(f"Δ = {e_mean - r_mean:+.0f} pts",
            xy=(len(eps)*0.42, (r_mean + e_mean)/2),
            color=AMBER, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=AMBER, alpha=0.9))

ax.set_xlabel("Episode", fontsize=10)
ax.set_ylabel("Cumulative Episode Reward", fontsize=10)
ax.set_title("Episode Reward — Random vs Scripted Expert  (30 episodes, Tier 1)",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "baseline_vs_expert_episode_reward.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ baseline_vs_expert_episode_reward.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Agentic recall — bar + jitter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = dark_fig(11, 5.5)

r_recall = rnd["agentic_recall"]
e_recall = exp["agentic_recall"]
r_rmean  = np.mean(r_recall)
e_rmean  = np.mean(e_recall)

jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(eps))
ax.bar([1], [r_rmean], width=0.5, color=RED,   alpha=0.25, zorder=1)
ax.bar([2], [e_rmean], width=0.5, color=GREEN, alpha=0.25, zorder=1)
ax.scatter(np.ones(len(eps)) + jitter, r_recall,
           color=RED,   s=35, alpha=0.75, zorder=3)
ax.scatter(np.full(len(eps), 2) + jitter, e_recall,
           color=GREEN, s=35, alpha=0.85, zorder=3)

ax.errorbar([1], [r_rmean], yerr=[[np.std(r_recall)], [np.std(r_recall)]],
            fmt="none", color=RED,   linewidth=2, capsize=6)
ax.errorbar([2], [e_rmean], yerr=[[np.std(e_recall)], [np.std(e_recall)]],
            fmt="none", color=GREEN, linewidth=2, capsize=6)

ax.annotate(f"{r_rmean:.2f}", xy=(1, r_rmean + 0.03), ha="center",
            color=RED, fontsize=12, fontweight="bold")
ax.annotate(f"{e_rmean:.2f}", xy=(2, e_rmean + 0.02), ha="center",
            color=GREEN, fontsize=12, fontweight="bold")
ax.annotate(f"3.4× improvement", xy=(1.5, 0.55),
            ha="center", color=AMBER, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=AMBER, alpha=0.9))

ax.set_xticks([1, 2])
ax.set_xticklabels(["random_policy", "scripted_expert"], color=WHITE, fontsize=11)
ax.set_ylabel("Agentic Recall  (fraction of GT evidence touched)", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("Agentic Recall — Random vs Scripted Expert  (30 episodes, Tier 1)",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "baseline_vs_expert_agentic_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ baseline_vs_expert_agentic_recall.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CoT validity score
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = dark_fig(11, 5)

r_cot = rnd["cot_validity_score"]
e_cot = exp["cot_validity_score"]

ax.step(eps, r_cot, color=RED,   linewidth=2, where="mid", label="random_policy")
ax.step(eps, e_cot, color=GREEN, linewidth=2, where="mid", label="scripted_expert")
ax.fill_between(eps, r_cot, step="mid", alpha=0.12, color=RED)
ax.fill_between(eps, e_cot, step="mid", alpha=0.12, color=GREEN)

ax.set_ylim(-0.05, 1.15)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0 — no CoT", "0.5", "1 — valid CoT"], color=MUTED)
ax.set_xlabel("Episode", fontsize=10)
ax.set_title("Chain-of-Thought Validity Score per Episode",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
ax.annotate("Expert always emits\nvalid <think> blocks", xy=(5, 1.02),
            color=GREEN, fontsize=9, fontstyle="italic")
ax.annotate("Random never emits\n<think> blocks → −2 pts/step", xy=(5, -0.02),
            color=RED, fontsize=9, fontstyle="italic")
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "baseline_vs_expert_cot_validity_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ baseline_vs_expert_cot_validity_score.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Reward constants waterfall (from the article reward table)
# ─────────────────────────────────────────────────────────────────────────────
rewards = {
    "CASE_WON":          1000,
    "CASE_PARTIAL":       250,
    "CONTRADICTION":      100,
    "LINK_SHELL":          50,
    "EXTRACT_ENTITY":      10,
    "CODEACT_BONUS":        5,
    "NPI_EXACT_MATCH":     25,
    "DOC_CLAIM_MATCH":     30,
    "NPI_MISMATCH":       -20,
    "HALLUCINATED_LINK":  -20,
    "FORMAT_GATE":        -10,
    "COT_MISSING":         -2,
    "HALLUCINATED_ENTITY":-50,
}

fig, ax = dark_fig(12, 5.5)
labels = list(rewards.keys())
vals   = list(rewards.values())
colors = [GREEN if v > 0 else RED for v in vals]

bars = ax.barh(labels, vals, color=colors, alpha=0.82, edgecolor=GRID, linewidth=0.5)
ax.axvline(0, color=MUTED, linewidth=0.8)

for bar, val in zip(bars, vals):
    offset = 8 if val >= 0 else -8
    ha     = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+d}", va="center", ha=ha, color=WHITE, fontsize=9, fontweight="bold")

ax.set_xlabel("Point Value", fontsize=10)
ax.set_title("7-Layer RLVR Grader — Reward Constants",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
ax.tick_params(axis="y", labelsize=9, labelcolor=WHITE)
ax.set_xlim(-120, 1150)
pos_patch = mpatches.Patch(color=GREEN, alpha=0.8, label="Reward")
neg_patch = mpatches.Patch(color=RED,   alpha=0.8, label="Penalty")
ax.legend(handles=[pos_patch, neg_patch], facecolor=PANEL,
          edgecolor=GRID, labelcolor=WHITE, fontsize=9, loc="lower right")
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "reward_constants.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ reward_constants.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fraud typology multipliers — radar / polar bar
# ─────────────────────────────────────────────────────────────────────────────
typologies    = ["AKS\nViolation", "Dead\nPatient", "PPP\nFraud", "Shell\nChain",
                 "Foreign\nAffiliation", "Duplicate\nBill"]
multipliers   = [1.6, 1.0, 1.8, 2.0, 2.0, 1.0]
base_reward   = 100

fig = plt.figure(figsize=(8, 8), facecolor=BG)
ax  = fig.add_subplot(111, polar=True, facecolor=PANEL)
ax.set_facecolor(PANEL)

N      = len(typologies)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values = [m * base_reward for m in multipliers]
values_closed = values + [values[0]]
angles_closed = angles + [angles[0]]

ax.plot(angles_closed, values_closed, color=AMBER, linewidth=2)
ax.fill(angles_closed, values_closed, color=AMBER, alpha=0.18)
ax.scatter(angles, values, color=AMBER, s=80, zorder=5)

for angle, val, mult, label in zip(angles, values, multipliers, typologies):
    ax.annotate(f"{mult}×\n+{val:.0f}pts",
                xy=(angle, val + 14), ha="center", va="center",
                color=WHITE, fontsize=8.5, fontweight="bold")

ax.set_xticks(angles)
ax.set_xticklabels(typologies, color=WHITE, fontsize=9)
ax.set_yticks([50, 100, 150, 200])
ax.set_yticklabels(["50", "100", "150", "200"], color=MUTED, fontsize=7)
ax.tick_params(colors=MUTED)
ax.spines["polar"].set_color(GRID)
ax.grid(color=GRID, linewidth=0.5)
ax.set_ylim(0, 230)
ax.set_title("Fraud Typology Multipliers\n(base CONTRADICTION_REWARD = +100 pts)",
             color=WHITE, fontsize=11, fontweight="bold", pad=20)
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "typology_multipliers.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ typology_multipliers.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Investigation step-by-step reward accumulation (from the article demo)
# ─────────────────────────────────────────────────────────────────────────────
steps  = [0, 1, 2, 3, 4, 5, 6]
labels = ["Reset", "sql_query\n(corporate)", "query_corporate\n(Acme Shell)",
          "link_shell\n(E003→E002)", "claim_contradiction\n(duplicate_bill)",
          "claim_contradiction\n(dead_patient)", "submit_case"]
deltas  = [0, 1.9, 0.9, 51.9, 99.9, 100.9, 0]
cumulative = [sum(deltas[:i+1]) for i in range(len(deltas))]

fig, (ax1, ax2) = dark_fig2(14, 5.5)

bar_colors = [AMBER if d > 10 else CYAN if d > 0 else MUTED for d in deltas]
ax1.bar(steps, deltas, color=bar_colors, alpha=0.85, edgecolor=GRID, linewidth=0.5, width=0.6)
for i, (s, d) in enumerate(zip(steps, deltas)):
    if d > 0:
        ax1.text(s, d + 1.5, f"+{d:.1f}", ha="center", color=WHITE, fontsize=8.5, fontweight="bold")
ax1.set_xticks(steps)
ax1.set_xticklabels(labels, fontsize=7.5, color=WHITE)
ax1.set_ylabel("Step Reward (pts)", fontsize=10)
ax1.set_title("Reward per Step — Demo Investigation", color=WHITE, fontsize=11, fontweight="bold", pad=10)

ax2.plot(steps, cumulative, color=AMBER, linewidth=2.5, marker="o", markersize=7, zorder=3)
ax2.fill_between(steps, cumulative, alpha=0.12, color=AMBER)
for s, c in zip(steps, cumulative):
    ax2.annotate(f"{c:.0f}", xy=(s, c + 4), ha="center",
                 color=WHITE, fontsize=8.5, fontweight="bold")
ax2.set_xticks(steps)
ax2.set_xticklabels(labels, fontsize=7.5, color=WHITE)
ax2.set_ylabel("Cumulative Reward (pts)", fontsize=10)
ax2.set_title("Cumulative Reward Accumulation", color=WHITE, fontsize=11, fontweight="bold", pad=10)

fig.tight_layout(pad=1.8)
fig.savefig(ASSETS / "investigation_step_rewards.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ investigation_step_rewards.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Hallucination count distribution — random vs expert
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = dark_fig(10, 5)

r_hall = rnd["hallucination_count"]
e_hall = exp["hallucination_count"]
bins   = np.arange(-0.5, max(r_hall) + 1.5, 1)

ax.hist(r_hall, bins=bins, color=RED,   alpha=0.7, label="random_policy",   edgecolor=GRID)
ax.hist(e_hall, bins=bins, color=GREEN, alpha=0.7, label="scripted_expert", edgecolor=GRID)
ax.set_xlabel("Hallucinations per Episode", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("Hallucination Count Distribution  (30 episodes each)",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.annotate("Expert: 0\nhallucinations\nin all 30 eps",
            xy=(0.15, 28), color=GREEN, fontsize=9, fontweight="bold")
leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
fig.tight_layout(pad=1.5)
fig.savefig(ASSETS / "hallucination_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ hallucination_distribution.png")

print("\nAll charts written to assets/")
