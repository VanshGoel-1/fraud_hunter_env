"""Generate all UML diagrams for FraudHunterEnv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0F172A"
PANEL   = "#1E293B"
BORDER  = "#334155"
HEADER  = "#1D4ED8"   # class header blue
HEADER2 = "#7C3AED"   # purple for env classes
HEADER3 = "#065F46"   # green for data classes
HEADER4 = "#92400E"   # amber for config
AMBER   = "#F59E0B"
CYAN    = "#22D3EE"
RED     = "#F87171"
GREEN   = "#4ADE80"
WHITE   = "#F8FAFC"
MUTED   = "#94A3B8"
LABEL   = "#CBD5E1"
DASHED  = "#475569"

# ── helpers ───────────────────────────────────────────────────────────────────
def fig_dark(w, h):
    f = plt.figure(figsize=(w, h), facecolor=BG)
    return f

def ax_dark(f, rect=[0,0,1,1]):
    a = f.add_axes(rect, facecolor=BG)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.axis("off")
    return a

def uml_box(ax, x, y, w, h, title, attrs, methods,
            hdr_color=HEADER, text_size=7.2):
    """Draw a UML class box. Returns bottom-centre y coordinate."""
    line_h = 0.022
    pad = 0.008
    n_attr = len(attrs); n_meth = len(methods)
    total_h = line_h*1.6 + line_h*(n_attr+1) + line_h*(n_meth+1) + pad*4

    # background
    ax.add_patch(FancyBboxPatch((x, y-total_h), w, total_h,
        boxstyle="round,pad=0.003", facecolor=PANEL, edgecolor=BORDER, lw=1.0))

    # header
    hdr_h = line_h * 1.6
    ax.add_patch(FancyBboxPatch((x, y-hdr_h), w, hdr_h,
        boxstyle="round,pad=0.003", facecolor=hdr_color, edgecolor=BORDER, lw=1.0))
    ax.text(x + w/2, y - hdr_h/2, f"«class»\n{title}",
            ha="center", va="center", color=WHITE,
            fontsize=text_size+0.5, fontweight="bold", linespacing=1.3)

    # separator
    sep1_y = y - hdr_h
    ax.axhline(sep1_y, xmin=x, xmax=x+w, color=BORDER, lw=0.5)

    # attributes
    ay = sep1_y - pad
    for a in attrs:
        ax.text(x + pad*2, ay - line_h/2, a, ha="left", va="center",
                color=LABEL, fontsize=text_size, family="monospace")
        ay -= line_h
    sep2_y = ay - pad
    ax.plot([x, x+w], [sep2_y, sep2_y], color=BORDER, lw=0.5)

    # methods
    my = sep2_y - pad
    for m in methods:
        ax.text(x + pad*2, my - line_h/2, m, ha="left", va="center",
                color=CYAN, fontsize=text_size, family="monospace")
        my -= line_h

    return y - total_h   # bottom y

def arrow(ax, x1, y1, x2, y2, style="->", color=MUTED, lw=1.0,
          label="", dashed=False):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color,
                        lw=lw, linestyle=ls,
                        connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.005, my+0.005, label, color=AMBER,
                fontsize=6.5, ha="center", va="bottom")

def title_banner(ax, text, y=0.97):
    ax.text(0.5, y, text, ha="center", va="top", color=WHITE,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=HEADER, edgecolor=BORDER, lw=1))

# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASS DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(18, 13)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — Class Diagram")

# FraudHunterAction
uml_box(ax, 0.01, 0.92, 0.21, 0,
    "FraudHunterAction",
    ["+ kind: ActionKind",
     "+ think_trace: str",
     "+ entity_name: str | None",
     "+ npi_code: str | None",
     "+ sql_statement: str | None",
     "+ python_code: str | None",
     "+ pdf_path: str | None",
     "+ child_entity: str | None",
     "+ contradiction_kind: str | None"],
    ["+ _require_kind_fields()"],
    hdr_color=HEADER3)

# FraudHunterObservation
uml_box(ax, 0.24, 0.92, 0.21, 0,
    "FraudHunterObservation",
    ["+ case_brief: str | None",
     "+ tool_output: str | None",
     "+ base64_document: str | None",
     "+ grader_feedback: str | None",
     "+ step_count: int",
     "+ budget_remaining: int",
     "+ difficulty_tier: int",
     "+ reward: float",
     "+ done: bool",
     "+ info: dict | None"],
    [],
    hdr_color=HEADER3)

# FraudHunterEnvironment (central)
uml_box(ax, 0.47, 0.92, 0.23, 0,
    "FraudHunterEnvironment",
    ["- _bank_dir: Path",
     "- _case: CaseHandle",
     "- _rng: Random",
     "- _extracted: set",
     "- _linked: set",
     "- _contradictions: set",
     "- _submitted: bool",
     "- _difficulty_tier: int",
     "- _session_id: str",
     "- _diff_mgr: DifficultyManager"],
    ["+ reset() → Observation",
     "+ step(action) → Observation",
     "+ close()",
     "+ state() → dict"],
    hdr_color=HEADER2)

# DifficultyManager
uml_box(ax, 0.01, 0.52, 0.19, 0,
    "DifficultyManager",
    ["- _profiles: dict[str, AgentProfile]",
     "- _lock: threading.Lock"],
    ["+ get_tier(session_id) → int",
     "+ record_episode(session_id, r)",
     "+ get_stats(session_id) → dict"],
    hdr_color=HEADER4)

# AgentProfile
uml_box(ax, 0.22, 0.52, 0.19, 0,
    "AgentProfile",
    ["+ session_id: str",
     "+ reward_history: deque",
     "+ episode_count: int",
     "+ current_tier: int",
     "+ best_reward: float",
     "+ total_format_errors: int"],
    ["+ record_episode(reward, tier)",
     "+ rolling_avg() → float"],
    hdr_color=HEADER4)

# OnlineRLPolicy
uml_box(ax, 0.72, 0.92, 0.20, 0,
    "OnlineRLPolicy",
    ["+ learning_rate: float",
     "+ temperature: float",
     "- _arms: list[str]",
     "- _weights: dict[str, list]",
     "- _baseline: float",
     "- _updates: int",
     "- _pending: dict[str, PendingDecision]",
     "- _lock: threading.Lock"],
    ["+ choose_arm(obs, obj, allow_llm)",
     "+ update(token, reward)",
     "+ fallback_action(obs, obj)",
     "+ snapshot() → dict",
     "+ save(path)",
     "+ reset()"],
    hdr_color=HEADER)

# PendingDecision
uml_box(ax, 0.72, 0.52, 0.18, 0,
    "PendingDecision",
    ["+ chosen_arm: str",
     "+ probs: dict[str, float]",
     "+ features: list[float]",
     "+ created_at: float"],
    [],
    hdr_color=HEADER)

# CORSConfig
uml_box(ax, 0.43, 0.52, 0.18, 0,
    "CORSConfig",
    ["+ origins: list[str]",
     "+ allow_credentials: bool",
     "+ methods: list[str]",
     "+ headers: list[str]"],
    ["+ from_env() → CORSConfig",
     "+ register(app: FastAPI)"],
    hdr_color=HEADER4)

# FraudHunterEnv (client)
uml_box(ax, 0.01, 0.20, 0.19, 0,
    "FraudHunterEnv\n(client)",
    ["+ base_url: str",
     "(inherits EnvClient)"],
    ["+ reset() → Observation",
     "+ step(action) → tuple",
     "+ _step_payload(action)",
     "+ _parse_result(resp)"],
    hdr_color=HEADER3)

# ── relationships ──────────────────────────────────────────────────────────────
# FraudHunterEnvironment uses Action → Observation
arrow(ax, 0.32, 0.83, 0.47, 0.85, style="-|>", color=CYAN, lw=1.2,
      label="step(action) →")
# Environment has DifficultyManager
arrow(ax, 0.47, 0.72, 0.20, 0.52, style="-|>", color=AMBER, lw=1.0,
      label="has-a")
# DifficultyManager has AgentProfile
arrow(ax, 0.20, 0.42, 0.31, 0.42, style="->", color=AMBER, lw=1.0,
      label="1..*")
# OnlineRLPolicy has PendingDecision
arrow(ax, 0.81, 0.72, 0.81, 0.52, style="->", color=CYAN, lw=1.0,
      label="0..*")
# client uses Action
arrow(ax, 0.10, 0.20, 0.10, 0.55, style="->", color=MUTED, lw=0.8,
      dashed=True, label="uses")
# FraudHunterEnv → FraudHunterObservation
arrow(ax, 0.20, 0.19, 0.35, 0.77, style="->", color=MUTED, lw=0.8,
      dashed=True)

# legend
lx, ly = 0.73, 0.35
ax.text(lx, ly+0.05, "Legend", color=WHITE, fontsize=8, fontweight="bold")
for sym, desc, col in [
    ("---->", "association / uses",  CYAN),
    ("-  ->", "dependency (dashed)", MUTED),
    ("---->", "composition",         AMBER),
]:
    ax.text(lx, ly, f"{sym}  {desc}", color=col, fontsize=7.5, family="monospace")
    ly -= 0.04

f.tight_layout(pad=0.3)
f.savefig(ASSETS / "uml_class_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_class_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SEQUENCE DIAGRAM — reset/step flow
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(16, 12)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — Sequence Diagram  (Episode Lifecycle)")

participants = ["Agent / LLM", "FastAPI\nServer", "FraudHunter\nEnvironment",
                "RLVR\nGrader", "Case DB\n(SQLite)"]
px = [0.08, 0.25, 0.45, 0.65, 0.83]
lw_life = 0.8

# lifeline headers
for x, p in zip(px, participants):
    ax.add_patch(FancyBboxPatch((x-0.06, 0.88), 0.12, 0.06,
        boxstyle="round,pad=0.005", facecolor=HEADER, edgecolor=BORDER, lw=1))
    ax.text(x, 0.91, p, ha="center", va="center", color=WHITE,
            fontsize=8.5, fontweight="bold", linespacing=1.3)

# lifelines
for x in px:
    ax.plot([x, x], [0.88, 0.04], color=DASHED, lw=1.0, linestyle=(0,(6,3)))

# activation boxes
def actbox(x, y_top, y_bot, color=HEADER):
    ax.add_patch(mpatches.Rectangle((x-0.008, y_bot), 0.016, y_top-y_bot,
        facecolor=color, edgecolor=BORDER, lw=0.5, alpha=0.7))

# sequence messages
msgs = [
    # (from_x_idx, to_x_idx, y, label, reply, note)
    (0, 1, 0.84, "POST /reset", False, ""),
    (1, 2, 0.80, "reset()", False, ""),
    (2, 4, 0.76, "load_case(seed)", False, ""),
    (4, 2, 0.72, "CaseHandle + SQLite conn", True, ""),
    (2, 3, 0.68, "init_grader_state()", False, ""),
    (3, 2, 0.64, "GraderState", True, ""),
    (2, 1, 0.60, "FraudHunterObservation", True, ""),
    (1, 0, 0.56, "{ observation, reward=0, done=false }", True, ""),
    (0, 0, 0.50, "LLM generates FraudHunterAction", False, "«think»"),
    (0, 1, 0.46, "POST /step  { action: {...} }", False, ""),
    (1, 2, 0.42, "step(FraudHunterAction)", False, ""),
    (2, 3, 0.38, "grade(action, case, state)", False, ""),
    (3, 4, 0.34, "verify ground_truth", False, ""),
    (4, 3, 0.30, "GT rows", True, ""),
    (3, 2, 0.26, "GraderOutput(reward, feedback, done)", True, ""),
    (2, 1, 0.22, "FraudHunterObservation(reward, done)", True, ""),
    (1, 0, 0.18, "{ observation, reward, done }", True, ""),
    (0, 0, 0.13, "if done → new episode; else → next step", False, "loop"),
]

for mi, (fi, ti, y, label, is_reply, note) in enumerate(msgs):
    x1, x2 = px[fi], px[ti]
    col = MUTED if is_reply else WHITE
    ls  = (0,(4,2)) if is_reply else "solid"
    style = "<-" if is_reply and fi != ti else "->"
    if fi == ti:  # self-message (loop/note)
        ax.annotate("", xy=(x1+0.06, y-0.015), xytext=(x1, y),
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0,
                            connectionstyle="arc3,rad=-0.4"))
        ax.text(x1+0.07, y-0.008, label, color=AMBER, fontsize=7.5,
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL,
                          edgecolor=AMBER, lw=0.7))
    else:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle=style, color=col, lw=0.9,
                            linestyle=ls))
        mx = (x1+x2)/2
        ax.text(mx, y+0.012, label, ha="center", va="bottom",
                color=col, fontsize=7.5)
        if note:
            ax.text(max(x1,x2)+0.015, y, f"«{note}»",
                    color=AMBER, fontsize=6.5, va="center", style="italic")

# activation boxes on server/env/grader during active processing
actbox(px[1], 0.84, 0.14)
actbox(px[2], 0.80, 0.14, color=HEADER2)
actbox(px[3], 0.68, 0.24, color=HEADER4)
actbox(px[4], 0.76, 0.28, color=HEADER3)

# divider between reset and step phases
ax.plot([0.03, 0.95], [0.53, 0.53], color=BORDER, lw=0.8, linestyle="--")
ax.text(0.5, 0.535, "─── episode step (repeats until done=true) ───",
        ha="center", va="bottom", color=DASHED, fontsize=7.5, style="italic")

f.savefig(ASSETS / "uml_sequence_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_sequence_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPONENT DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(16, 11)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — Component Diagram")

def comp_box(ax, x, y, w, h, title, items, color=HEADER, dashed=False):
    ls = (0,(5,3)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.01", facecolor=PANEL,
        edgecolor=color, lw=1.5, linestyle=ls))
    ax.add_patch(FancyBboxPatch((x, y+h-0.055), w, 0.055,
        boxstyle="round,pad=0.005", facecolor=color,
        edgecolor=color, lw=0))
    ax.text(x+w/2, y+h-0.027, title, ha="center", va="center",
            color=WHITE, fontsize=8.5, fontweight="bold")
    for i, item in enumerate(items):
        ax.text(x+0.012, y+h-0.075-i*0.038, f"▸ {item}",
                va="top", color=LABEL, fontsize=7.5)

def port(ax, x, y, r=0.008, color=AMBER):
    ax.add_patch(plt.Circle((x, y), r, facecolor=color,
                             edgecolor=BG, lw=0.5, zorder=5))

# Agent
comp_box(ax, 0.01, 0.70, 0.18, 0.18, "Agent / LLM Policy",
    ["Llama 3.2-11B VLM", "GRPO LoRA adapter",
     "generate FraudHunterAction", "emit <think> trace"], color=HEADER2)

# FastAPI Server
comp_box(ax, 0.26, 0.55, 0.30, 0.33, "FastAPI Server  (server/app.py)",
    ["POST /reset   POST /step", "GET  /health  GET /metrics",
     "POST /fraud_hunter/agent_action",
     "POST /fraud_hunter/nl_action",
     "RateLimitMiddleware (token bucket)",
     "APIKeyMiddleware", "CORSConfig",
     "WebSocket /ws (OpenEnv protocol)"], color=HEADER)

# Environment
comp_box(ax, 0.26, 0.10, 0.30, 0.40, "FraudHunterEnvironment",
    ["reset() / step() / state()",
     "CodeAct Sandbox (5s timeout)",
     "7-Layer RLVR Grader",
     "DifficultyManager (RLVE)",
     "OCR: Tesseract + pdfplumber",
     "NPI Luhn validator"], color=HEADER2)

# Case Bank
comp_box(ax, 0.62, 0.55, 0.22, 0.33, "Case Bank  (data/case_bank/)",
    ["tier_1/ … tier_5/",
     "medicare_records.db (SQLite)",
     "scanned_claims/*.pdf",
     "intercepted_comms/*.txt",
     "CMS SynPUF schema"], color=HEADER3)

# Online RL
comp_box(ax, 0.62, 0.10, 0.22, 0.40, "Online RL  (server/online_rl.py)",
    ["OnlineRLPolicy (bandit)",
     "arms: llm / sql / ocr / comms",
     "contextual features (10-dim)",
     "REINFORCE weight update",
     "save/load bandit_weights.json",
     "agent_action_online endpoint"], color=HEADER)

# Training
comp_box(ax, 0.01, 0.10, 0.20, 0.42, "Training  (training/)",
    ["grpo_train.ipynb",
     "GRPO + DAPO loss",
     "clip-higher ε=0.28",
     "β=0 (zero KL)",
     "train_api.py",
     "API-in-the-loop REINFORCE",
     "plot_train_api.py"], color=HEADER4)

# Web UI
comp_box(ax, 0.62, 0.88, 0.22, 0.08, "Web Dashboard  (web/index.html)",
    ["SSE metrics stream / leaderboard"], color=DASHED, dashed=True)

# ports / connections
port(ax, 0.19, 0.795)   # agent → server
port(ax, 0.26, 0.795)
arrow(ax, 0.198, 0.795, 0.252, 0.795, color=CYAN, lw=1.5,
      label="HTTP / WebSocket")

port(ax, 0.56, 0.71)    # server → env
port(ax, 0.56, 0.48)
arrow(ax, 0.56, 0.548, 0.56, 0.488, color=AMBER, lw=1.5,
      label="step/reset")

port(ax, 0.62, 0.30)    # env → case bank
arrow(ax, 0.56, 0.30, 0.612, 0.30, color=GREEN, lw=1.5,
      label="load/query")

port(ax, 0.62, 0.25)    # env → online rl (feedback)
arrow(ax, 0.56, 0.22, 0.612, 0.22, color=AMBER, lw=1.5, dashed=True,
      label="reward signal")

port(ax, 0.21, 0.30)    # training → env (episodes)
arrow(ax, 0.21, 0.30, 0.252, 0.30, color=HEADER4, lw=1.5, dashed=True,
      label="train episodes")

port(ax, 0.73, 0.55)    # online rl ↔ case bank (weights)
arrow(ax, 0.73, 0.55, 0.73, 0.498, color=MUTED, lw=1.0, dashed=True)

f.savefig(ASSETS / "uml_component_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_component_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ER DIAGRAM — SQLite case bank schema
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(18, 13)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — Entity-Relationship Diagram  (CMS SynPUF Schema)")

def er_table(ax, x, y, name, pk_cols, cols, color=HEADER):
    row_h = 0.028; pad = 0.006
    all_cols = pk_cols + cols
    h = row_h * 1.5 + row_h * len(all_cols) + pad * 2
    ax.add_patch(FancyBboxPatch((x, y-h), 0.23, h,
        boxstyle="round,pad=0.004", facecolor=PANEL, edgecolor=color, lw=1.2))
    # header
    ax.add_patch(FancyBboxPatch((x, y-row_h*1.5), 0.23, row_h*1.5,
        boxstyle="round,pad=0.004", facecolor=color, edgecolor=color, lw=0))
    ax.text(x+0.115, y-row_h*0.75, name, ha="center", va="center",
            color=WHITE, fontsize=8, fontweight="bold")
    # divider
    ax.plot([x, x+0.23], [y-row_h*1.5]*2, color=color, lw=0.7)
    cy = y - row_h*1.5 - pad
    for i, col in enumerate(all_cols):
        is_pk = col in pk_cols
        prefix = "🔑 " if is_pk else "   "
        fc = AMBER if is_pk else LABEL
        ax.text(x+0.008, cy - row_h/2, f"{prefix}{col}", ha="left", va="center",
                color=fc, fontsize=7, family="monospace")
        cy -= row_h
    return y - h   # bottom y

# beneficiary_summary
er_table(ax, 0.01, 0.92, "beneficiary_summary",
    ["DESYNPUF_ID"],
    ["BENE_BIRTH_DT", "BENE_DEATH_DT", "BENE_SEX_IDENT_CD",
     "BENE_RACE_CD", "BENE_ESRD_IND", "SP_ALZHDMTA",
     "BENE_HI_CVRAGE_TOT_MOS"], HEADER3)

# carrier_claims
er_table(ax, 0.26, 0.92, "carrier_claims",
    ["CLM_ID"],
    ["DESYNPUF_ID", "CLM_FROM_DT", "CLM_THRU_DT",
     "PRVDR_NPI", "HCFASPCL_CD",
     "LINE_ICD9_DGNS_CD", "LINE_NCH_PMT_AMT",
     "HCPCS_CD"], HEADER3)

# corporate_registry
er_table(ax, 0.51, 0.92, "corporate_registry",
    ["entity_id"],
    ["entity_name", "tax_id", "parent_entity_id",
     "ubo_id", "state", "registered_date",
     "npi", "tier"], HEADER2)

# ground_truth
er_table(ax, 0.76, 0.92, "ground_truth",
    ["id (rowid)"],
    ["kind  ← entity | shell_link",
     "       | contradiction",
     "payload_json (JSON)"], HEADER)

# case_metadata
er_table(ax, 0.01, 0.55, "case_metadata",
    ["key"],
    ["value", "← tier, seed, case_id,",
     "  typologies (JSON array)"], HEADER4)

# loan_applications
er_table(ax, 0.26, 0.55, "loan_applications",
    ["loan_id"],
    ["entity_id", "program", "loan_amount",
     "claimed_employees", "period_start",
     "period_end"], HEADER4)

# general_ledger
er_table(ax, 0.51, 0.55, "general_ledger",
    ["tx_id"],
    ["tx_date", "debit_account",
     "credit_account", "amount",
     "memo"], HEADER4)

# evidence_documents
er_table(ax, 0.76, 0.55, "evidence_documents",
    ["doc_id"],
    ["claim_id", "pdf_path",
     "tier", "created_at"], HEADER3)

# payroll_records
er_table(ax, 0.01, 0.28, "payroll_records",
    ["payroll_id"],
    ["entity_id", "period_end",
     "employee_count", "total_wages"], HEADER4)

# prescription_drug_events
er_table(ax, 0.26, 0.28, "prescription_drug_events",
    ["PDE_ID"],
    ["DESYNPUF_ID", "PRVDR_NPI",
     "SRVC_DT", "PROD_SRVC_ID",
     "TOT_RX_CST_AMT"], HEADER3)

# inpatient_claims
er_table(ax, 0.51, 0.28, "inpatient_claims",
    ["CLM_ID"],
    ["DESYNPUF_ID", "CLM_FROM_DT",
     "CLM_THRU_DT", "PRVDR_NPI",
     "CLM_PMT_AMT"], HEADER3)

# outpatient_claims
er_table(ax, 0.76, 0.28, "outpatient_claims",
    ["CLM_ID"],
    ["DESYNPUF_ID", "CLM_FROM_DT",
     "CLM_THRU_DT", "PRVDR_NPI",
     "CLM_PMT_AMT"], HEADER3)

# ── FK relationships ───────────────────────────────────────────────────────────
fk_arrows = [
    # (x1, y1, x2, y2, label)
    (0.235, 0.835, 0.26,  0.835, "DESYNPUF_ID"),
    (0.235, 0.145, 0.26,  0.175, "DESYNPUF_ID"),
    (0.51,  0.79,  0.49,  0.79,  "parent_entity_id"),
    (0.49,  0.79,  0.49,  0.835, ""),
    (0.26,  0.45,  0.51,  0.68,  "entity_id"),
    (0.51,  0.45,  0.26,  0.455, ""),
    (0.735, 0.45,  0.62,  0.76,  "claim_id → CLM_ID"),
]
for x1, y1, x2, y2, lbl in fk_arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=0.8,
                        connectionstyle="arc3,rad=0.0"))
    if lbl:
        ax.text((x1+x2)/2+0.005, (y1+y2)/2+0.005, lbl,
                color=AMBER, fontsize=6, ha="center")

f.savefig(ASSETS / "uml_er_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_er_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STATE MACHINE DIAGRAM — episode lifecycle
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(14, 10)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — Episode State Machine")

def state_node(ax, x, y, r, label, sublabel="", color=HEADER, text_size=9):
    ax.add_patch(plt.Circle((x, y), r, facecolor=color,
                             edgecolor=WHITE, lw=1.5, zorder=3))
    ax.text(x, y + (0.012 if sublabel else 0), label, ha="center", va="center",
            color=WHITE, fontsize=text_size, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y-0.022, sublabel, ha="center", va="center",
                color=LABEL, fontsize=7, zorder=4)

def state_arrow(ax, x1, y1, x2, y2, label, color=MUTED, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                        connectionstyle=f"arc3,rad={rad}"))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my+0.02, label, ha="center", va="bottom",
            color=color, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                      edgecolor=color, lw=0.5, alpha=0.85))

# states
state_node(ax, 0.50, 0.90, 0.018, "●", color="#1E293B")          # initial
state_node(ax, 0.50, 0.75, 0.060, "IDLE",
           "server ready", BORDER)
state_node(ax, 0.50, 0.52, 0.070, "INVESTIGATING",
           "step_count < 60\ndone = False", HEADER2)
state_node(ax, 0.20, 0.25, 0.065, "FORMAT\nERROR",
           "done = True\nreward −10", RED)
state_node(ax, 0.50, 0.15, 0.065, "BUDGET\nEXHAUSTED",
           "step_count = 60\ndone = True", HEADER4)
state_node(ax, 0.80, 0.25, 0.065, "CASE\nSUBMITTED",
           "done = True\nreward ±", GREEN)
state_node(ax, 0.50, 0.02, 0.018, "◎", color=HEADER3)            # final

# transitions
state_arrow(ax, 0.50, 0.882, 0.50, 0.812, "server start", MUTED)
state_arrow(ax, 0.50, 0.690, 0.50, 0.592, "POST /reset\nenv.reset()", CYAN)
state_arrow(ax, 0.50, 0.452, 0.50, 0.452, "", AMBER)   # placeholder

# self-loop on INVESTIGATING
ax.annotate("", xy=(0.57, 0.55), xytext=(0.57, 0.49),
    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5,
                    connectionstyle="arc3,rad=-0.7"))
ax.text(0.665, 0.52, "POST /step\nvalid action\n(not terminal)",
        ha="center", va="center", color=AMBER, fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                  edgecolor=AMBER, lw=0.5, alpha=0.85))

# → FORMAT ERROR
state_arrow(ax, 0.437, 0.475, 0.258, 0.308, "invalid JSON\nformat_gate", RED, rad=0.1)
# → BUDGET EXHAUSTED
state_arrow(ax, 0.50,  0.450, 0.50,  0.215, "step_count\n= 60", HEADER4, rad=0.0)
# → CASE SUBMITTED
state_arrow(ax, 0.563, 0.475, 0.742, 0.308, "submit_case\naction", GREEN, rad=-0.1)

# all terminal → reset
state_arrow(ax, 0.165, 0.198, 0.165, 0.80,  "", MUTED, rad=0)
ax.plot([0.165, 0.43], [0.80, 0.80], color=MUTED, lw=1.2, linestyle=(0,(4,2)))
ax.plot([0.165, 0.165],[0.198, 0.80], color=MUTED, lw=1.2, linestyle=(0,(4,2)))
ax.text(0.12, 0.50, "POST /reset\n(new episode)", color=MUTED, fontsize=7,
        ha="center", style="italic")

state_arrow(ax, 0.50, 0.085, 0.50, 0.038, "process exits", MUTED)
state_arrow(ax, 0.835, 0.198, 0.835, 0.80, "", MUTED, rad=0)
ax.plot([0.57, 0.835],[0.80, 0.80], color=MUTED, lw=1.2, linestyle=(0,(4,2)))
ax.plot([0.835,0.835],[0.198,0.80], color=MUTED, lw=1.2, linestyle=(0,(4,2)))

# reward annotations inside states
for x, y, txt, col in [
    (0.80, 0.195, "+250 partial\n+1000 case won", GREEN),
    (0.20, 0.220, "−10 format gate\nepisode ends", RED),
    (0.50, 0.120, "step decay −0.1/step\nepisode ends", AMBER),
]:
    ax.text(x, y, txt, ha="center", va="top", color=col, fontsize=6.5)

f.savefig(ASSETS / "uml_state_machine.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_state_machine.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ACTIVITY DIAGRAM — 7-layer RLVR grader
# ══════════════════════════════════════════════════════════════════════════════
f = fig_dark(12, 16)
ax = ax_dark(f)
title_banner(ax, "FraudHunterEnv — 7-Layer RLVR Grader  (Activity Diagram)", y=0.985)

layers = [
    ("Layer 1\nFormat Gate",    "Valid Pydantic schema?",  "−10 + terminate",  HEADER,  False),
    ("Layer 2\nCoT Validity",   "Has <think> block?",      "−2 pts",           HEADER,  True),
    ("Layer 3\nStep Decay",     "Apply −0.1 per step",     "",                 HEADER4, False),
    ("Layer 4\nEntity Extract", "Entity in ground truth?", "−50 hallucination",HEADER2, True),
    ("Layer 5\nShell Link",     "Pair in GT shell_link?",  "−20 wrong link",   HEADER2, True),
    ("Layer 6\nNPI Luhn",       "Luhn + registry match?",  "−20 mismatch",     HEADER,  True),
    ("Layer 7\nContradiction",  "In GT contradiction?",    "No match → 0",     HEADER2, True),
]

rewards = [
    "−10, done",
    "+1 grounded\n−0.005/word > 50",
    "always −0.1",
    "+10 correct\n−50 halluc.",
    "+50 confirmed\n−20 wrong",
    "+25 exact match\n−20 fail",
    "+100×multiplier\n+1000 if all GT",
]

start_y = 0.95
box_h   = 0.072
gap     = 0.025
x_main  = 0.18
w_main  = 0.42
x_rew   = 0.65
w_rew   = 0.28

# start dot
ax.add_patch(plt.Circle((x_main + w_main/2, start_y + 0.01), 0.015,
                          facecolor=WHITE, edgecolor=WHITE, zorder=5))
ax.plot([x_main+w_main/2]*2, [start_y+0.025, start_y],
        color=MUTED, lw=1.5)

cy = start_y
for i, ((lname, question, neg, color, has_branch), reward_txt) in \
        enumerate(zip(layers, rewards)):
    # main box
    ax.add_patch(FancyBboxPatch((x_main, cy - box_h), w_main, box_h,
        boxstyle="round,pad=0.007", facecolor=color, edgecolor=WHITE, lw=1.2))
    ax.text(x_main + w_main/2, cy - box_h*0.35,
            lname.replace("\n", " — "), ha="center", va="center",
            color=WHITE, fontsize=9, fontweight="bold")
    ax.text(x_main + w_main/2, cy - box_h*0.70,
            question, ha="center", va="center", color=LABEL, fontsize=7.5)

    # reward box
    ax.add_patch(FancyBboxPatch((x_rew, cy - box_h + 0.008), w_rew, box_h-0.016,
        boxstyle="round,pad=0.006", facecolor=PANEL,
        edgecolor=AMBER, lw=0.8, linestyle=(0,(4,2))))
    ax.text(x_rew + w_rew/2, cy - box_h/2, reward_txt,
            ha="center", va="center", color=AMBER, fontsize=7.5,
            linespacing=1.4)

    # connector from main box to reward
    ax.annotate("", xy=(x_rew, cy-box_h/2),
                xytext=(x_main+w_main, cy-box_h/2),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8))

    if has_branch and neg:
        ax.text(x_main+w_main+0.005, cy-box_h/2+0.005,
                "YES", color=GREEN, fontsize=7, fontweight="bold")

    # negative branch label
    if neg:
        ax.text(x_main - 0.005, cy - box_h/2, f"NO →",
                ha="right", va="center", color=RED, fontsize=7, fontweight="bold")

    # down arrow
    next_y = cy - box_h - gap
    if i < len(layers)-1:
        ax.annotate("", xy=(x_main+w_main/2, next_y+0.002),
                    xytext=(x_main+w_main/2, cy-box_h),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    cy = next_y

# end diamond / submission
ax.add_patch(mpatches.FancyArrow(x_main+w_main/2, cy+0.01, 0, -0.03,
    width=0.001, head_width=0.02, head_length=0.015,
    facecolor=MUTED, edgecolor=MUTED))

# end node
end_y = cy - 0.04
ax.add_patch(plt.Circle((x_main+w_main/2, end_y), 0.022,
    facecolor=HEADER3, edgecolor=WHITE, lw=1.5, zorder=5))
ax.text(x_main+w_main/2, end_y, "Return\nObservation",
        ha="center", va="center", color=WHITE, fontsize=7, fontweight="bold")

# multiplier callout
ax.text(0.01, 0.12,
    "Typology multipliers:\n"
    "  foreign_affiliation  × 2.0\n"
    "  ppp_fraud            × 1.8\n"
    "  aks_violation        × 1.6\n"
    "  + PDF chain          × 1.5\n"
    "  dead_patient         × 1.0",
    va="top", color=LABEL, fontsize=7.5, family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
              edgecolor=AMBER, lw=0.8))

f.savefig(ASSETS / "uml_grader_activity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uml_grader_activity.png")

print(f"\nAll 6 UML diagrams saved to {ASSETS}")
