"""Plot reward curve from assets/train_api_log.jsonl → assets/train_api_curves.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
LOG   = _REPO / "assets" / "train_api_log.jsonl"
OUT   = _REPO / "assets" / "train_api_curves.png"
ARMS  = ["sql_schema", "ocr_doc", "shell_trace", "contradiction", "submit"]
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
if not rows:
    raise SystemExit(f"No data in {LOG}")

eps      = [r["episode"]  for r in rows]
rewards  = [r["reward"]   for r in rows]
baseline = [r["baseline"] for r in rows]

# Smoothed reward (window=10)
w = min(10, len(rewards))
smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

# Left: reward curve
ax1.plot(eps, rewards, alpha=0.3, color="#aec7e8", linewidth=0.8, label="raw")
ax1.plot(eps[w - 1:], smooth, color="#1f77b4", linewidth=2, label=f"smoothed (w={w})")
ax1.plot(eps, baseline, color="C3", linewidth=1.5, linestyle="--", label="baseline")
ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax1.set_xlabel("episode")
ax1.set_ylabel("reward")
ax1.set_title("API-in-the-loop REINFORCE — reward")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Right: arm probability over time
ax2.set_prop_cycle(color=COLORS)
for arm in ARMS:
    probs = [r["probs"].get(arm, 0.0) for r in rows]
    ax2.plot(eps, probs, linewidth=1.5, label=arm)
ax2.set_xlabel("episode")
ax2.set_ylabel("probability")
ax2.set_title("Bandit arm probabilities")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
