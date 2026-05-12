from __future__ import annotations

import json
import math
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class PendingDecision:
    chosen_arm: str
    probs: dict[str, float]
    features: list[float]
    created_at: float


class OnlineRLPolicy:
    """Lightweight contextual policy-gradient head for real-time online updates.

    This updates in-process weights on every feedback event while keeping the
    serving latency low. It intentionally trains a compact decision head over
    action strategies instead of fine-tuning full LLM weights online.
    """

    def __init__(self, learning_rate: float = 0.03, temperature: float = 1.0):
        self.learning_rate = float(max(1e-5, learning_rate))
        self.temperature = float(max(0.1, temperature))
        self._arms = ["llm", "sql_schema", "code_list_comms", "ocr_doc"]
        self._feature_dim = 10
        self._weights: dict[str, list[float]] = {
            arm: [0.0 for _ in range(self._feature_dim)] for arm in self._arms
        }
        self._baseline = 0.0
        self._updates = 0
        self._selection_count = {arm: 0 for arm in self._arms}
        self._pending: dict[str, PendingDecision] = {}
        self._lock = threading.Lock()

    def _dot(self, w: list[float], x: list[float]) -> float:
        return sum(a * b for a, b in zip(w, x, strict=True))

    def _softmax(self, logits: dict[str, float]) -> dict[str, float]:
        if not logits:
            return {}
        m = max(logits.values())
        exps = {k: math.exp((v - m) / self.temperature) for k, v in logits.items()}
        denom = sum(exps.values())
        if denom <= 0:
            uniform = 1.0 / float(len(logits))
            return {k: uniform for k in logits}
        return {k: (v / denom) for k, v in exps.items()}

    def _extract_ids(self, observation: dict[str, Any]) -> tuple[str | None, str | None]:
        text = " ".join(
            [
                str(observation.get("case_brief") or ""),
                str(observation.get("tool_output") or ""),
                str((observation.get("info") or {}).get("case_id") or ""),
            ]
        )
        bene_match = re.search(r"\b(BENE[_-]?\d{1,6}|DESYNPUF_ID)\b", text)
        claim_match = re.search(r"\b(C\d{1,6}|CLM[_-]?\d{1,10})\b", text)
        bene = bene_match.group(1) if bene_match else None
        claim = claim_match.group(1) if claim_match else None
        return bene, claim

    def _build_features(self, observation: dict[str, Any], objective: str) -> list[float]:
        info = observation.get("info") or {}
        step_count = float(observation.get("step_count") or 0.0)
        budget = float(observation.get("budget_remaining") or 0.0)
        tier = float(observation.get("difficulty_tier") or 1.0)
        tool_output = str(observation.get("tool_output") or "")
        lower_objective = (objective or "").lower()
        return [
            1.0,
            min(step_count / 59.0, 1.0),
            min(max(budget, 0.0) / 100.0, 1.0),
            min(max(tier, 1.0), 5.0) / 5.0,
            1.0 if tool_output else 0.0,
            1.0 if "ocr" in lower_objective or "document" in lower_objective else 0.0,
            1.0 if "sql" in lower_objective or "table" in lower_objective else 0.0,
            1.0 if "entity" in lower_objective else 0.0,
            1.0 if "contradiction" in lower_objective or "fraud" in lower_objective else 0.0,
            1.0 if (info.get("replay_hash") is not None) else 0.0,
        ]

    def _template_action(self, arm: str, observation: dict[str, Any], objective: str) -> dict[str, Any]:
        bene_id, claim_id = self._extract_ids(observation)
        if arm == "sql_schema":
            return {
                "kind": "sql_query",
                "sql_statement": "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                "think_trace": "<think>Probe the database schema to ground the next evidence action.</think>",
            }
        if arm == "code_list_comms":
            return {
                "kind": "code_act",
                "python_code": "files = listdir('intercepted_comms')\nprint(files[:10])",
                "think_trace": "<think>Inspect intercepted communications to identify shell entities and links.</think>",
            }
        if arm == "ocr_doc":
            import os
            case_dir = (observation.get("info") or {}).get("case_dir", "")
            pdf_path = "scanned_claims/doc_claim.pdf"
            if case_dir:
                sc_dir = os.path.join(case_dir, "scanned_claims")
                try:
                    pdfs = sorted(f for f in os.listdir(sc_dir) if f.endswith(".pdf"))
                    if pdfs:
                        pdf_path = "scanned_claims/" + pdfs[0]
                except OSError:
                    pass
            return {
                "kind": "ocr_document",
                "pdf_path": pdf_path,
                "think_trace": "<think>Extract claim evidence from the scanned document for contradiction checks.</think>",
            }
        if bene_id or claim_id:
            return {
                "kind": "query_medicare",
                "beneficiary_id": bene_id,
                "claim_id": claim_id,
                "think_trace": "<think>Query Medicare records for concrete IDs before escalating to contradiction claims.</think>",
            }
        return {
            "kind": "sql_query",
            "sql_statement": "SELECT * FROM beneficiary_summary LIMIT 5",
            "think_trace": "<think>Sample beneficiary records to discover concrete IDs for targeted follow-up actions.</think>",
        }

    def fallback_action(self, observation: dict[str, Any], objective: str) -> dict[str, Any]:
        """Return a deterministic schema-valid action without mutating policy state."""
        return self._template_action("sql_schema", observation, objective)

    def choose_arm(
        self,
        observation: dict[str, Any],
        objective: str,
        allow_llm: bool,
    ) -> dict[str, Any]:
        features = self._build_features(observation, objective)
        with self._lock:
            active_arms = list(self._arms)
            if not allow_llm:
                active_arms = [a for a in active_arms if a != "llm"]
            logits = {arm: self._dot(self._weights[arm], features) for arm in active_arms}
            probs = self._softmax(logits)
            if not probs:
                raise RuntimeError("no available policy arms")
            r = random.random()
            accum = 0.0
            chosen = active_arms[-1]
            for arm in active_arms:
                accum += probs.get(arm, 0.0)
                if r <= accum:
                    chosen = arm
                    break
            token = str(uuid.uuid4())
            self._pending[token] = PendingDecision(
                chosen_arm=chosen,
                probs=probs,
                features=features,
                created_at=time.time(),
            )
            self._selection_count[chosen] = self._selection_count.get(chosen, 0) + 1

        payload = {
            "token": token,
            "arm": chosen,
            "probs": probs,
            "action": None if chosen == "llm" else self._template_action(chosen, observation, objective),
        }
        return payload

    def update(self, token: str, reward: float) -> dict[str, Any]:
        reward = float(reward)
        with self._lock:
            # keep memory bounded for stale decisions
            if len(self._pending) > 4096:
                cutoff = time.time() - 900
                stale = [k for k, v in self._pending.items() if v.created_at < cutoff]
                for key in stale:
                    self._pending.pop(key, None)

            decision = self._pending.pop(token, None)
            if decision is None:
                raise KeyError("unknown or expired decision token")

            x = decision.features
            chosen = decision.chosen_arm
            probs = decision.probs
            advantage = reward - self._baseline
            lr = self.learning_rate

            for arm in probs:
                coeff = (1.0 if arm == chosen else 0.0) - probs[arm]
                w = self._weights[arm]
                for i in range(self._feature_dim):
                    w[i] += lr * advantage * coeff * x[i]

            self._baseline = (0.95 * self._baseline) + (0.05 * reward)
            self._updates += 1

            return {
                "token": token,
                "chosen_arm": chosen,
                "reward": reward,
                "advantage": advantage,
                "baseline": self._baseline,
                "updates": self._updates,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "arms": list(self._arms),
                "baseline": self._baseline,
                "updates": self._updates,
                "selection_count": dict(self._selection_count),
                "weights": {k: list(v) for k, v in self._weights.items()},
                "pending_decisions": len(self._pending),
            }

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._weights = {
                arm: [0.0 for _ in range(self._feature_dim)] for arm in self._arms
            }
            self._baseline = 0.0
            self._updates = 0
            self._selection_count = {arm: 0 for arm in self._arms}
            self._pending.clear()
        return self.snapshot()
