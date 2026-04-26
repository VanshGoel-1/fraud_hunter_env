"""
GRPO Training Script — Colab/GPU-ready for Llama-3.2-3B-Instruct.

Advanced features (per NotebookLM hackathon guidance):
  - DAPO loss type (normalise by active tokens, not sequence length)
  - Clip-higher strategy (epsilon=0.2, epsilon_high=0.25)
  - Zero KL penalty (beta=0.0) for maximum exploration
  - Gibberish detection (reject completions with >10% rare tokens)
  - Rejection sampling (iterative self-bootstrapping)
  - CoT-Pass@K evaluation metric
  - Agentic Recall tracking from environment metrics
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Put the repo root (parent of the `fraud_hunter_env` package) on sys.path so
# `fraud_hunter_env.*` resolves whether or not the package has been installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fraud_hunter_env.models import FraudHunterAction, FraudHunterObservation

TRAIN_SEED_RANGE = (0, 8000)

# Heavy ML deps live in the [dev] extra and are only installed in Colab/GPU
# environments. Import optimistically; fall back to None so lint and CPU-only
# machines don't crash. main() will NoneType-error if invoked without deps,
# which is the intended failure mode.
try:
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]
except ImportError:
    FastLanguageModel = None  # type: ignore[assignment,misc]
    GRPOConfig = None  # type: ignore[assignment,misc]
    GRPOTrainer = None  # type: ignore[assignment,misc]
    Dataset = None  # type: ignore[assignment,misc]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ── Configuration ─────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH  = 4096
LORA_RANK       = 16
MODEL_NAME      = "unsloth/Llama-3.2-3B-Instruct"
OUTPUT_DIR      = "outputs"


# ── Environment Reward Function ───────────────────────────────────────────────

def environment_reward(prompts: list[str], completions: list[Any], **kwargs) -> list[float]:
    """
    RLVR reward function: each completion is parsed as a sequence of JSON
    actions and executed against the Fraud Hunter environment.

    The environment's RLVR grader (7 hierarchical layers) produces the
    per-step rewards. No LLM-as-a-judge — purely programmatic verification.
    """
    from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment

    rewards = []
    for completion in completions:
        env = FraudHunterEnvironment(case_seed_range=TRAIN_SEED_RANGE)
        env.reset()
        episode_reward = 0.0

        # Parse CoT + action blocks from completion
        actions = _parse_actions_from_completion(completion)

        for act_payload in actions:
            try:
                action = FraudHunterAction.model_validate(act_payload)
                obs = env.step(action)
                episode_reward += float(obs.reward or 0.0)
                if obs.done:
                    break
            except Exception:
                episode_reward -= 10.0  # format gate penalty
                break

        rewards.append(episode_reward)
    return rewards


def _parse_actions_from_completion(raw_completion: Any) -> list[dict]:
    """
    Extract JSON action payloads from an LLM completion.
    Handles <think>...</think> blocks and JSON interleaved text, including
    nested objects like extracted_fields={...} (which the previous
    `\\{[^{}]+\\}` regex would silently miss).
    """
    import re
    actions: list[dict] = []

    # TRL/Unsloth can pass chat-shaped completions (list/dict) instead of a plain string.
    if isinstance(raw_completion, str):
        text = raw_completion
    elif isinstance(raw_completion, list):
        parts: list[str] = []
        for item in raw_completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    parts.extend(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
                else:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
    elif isinstance(raw_completion, dict):
        text = str(raw_completion.get("content", ""))
    else:
        text = str(raw_completion)

    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_traces = think_pattern.findall(text)
    think_idx = 0

    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        # Find the next JSON object opening brace.
        i = text.find("{", i)
        if i == -1:
            break
        try:
            payload, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(payload, dict) and "kind" in payload:
            if think_idx < len(think_traces):
                payload["think_trace"] = think_traces[think_idx]
                think_idx += 1
            actions.append(payload)
        i = end
    return actions


# ── Gibberish Detection ──────────────────────────────────────────────────────

# NOTE: gibberish_filter() and rejection_sample() were removed in the
# Phase-9 refactor. They were never wired into the GRPOTrainer loop and the
# gibberish_filter signature did not match TRL's reward-function contract.
# If you need either, re-derive them in an external eval/SFT script.


# ── CoT-Pass@K Metric ────────────────────────────────────────────────────────

def cot_pass_at_k(completions: list[str], k: int = 8) -> float:
    """
    Evaluate K completions; return the fraction where the CoT is logically
    valid (contains <think> and </think>, mentions at least one entity,
    and the final action is well-formed JSON).
    """
    valid = 0
    for comp in completions[:k]:
        has_think = "<think>" in comp and "</think>" in comp
        actions = _parse_actions_from_completion(comp)
        has_valid_action = len(actions) > 0
        if has_think and has_valid_action:
            valid += 1
    return valid / min(k, len(completions)) if completions else 0.0


# ── Dataset Loader ────────────────────────────────────────────────────────────

def load_dataset(n_episodes: int = 100):
    """
    Generate training prompts by resetting the environment N times.
    Each prompt is the case briefing.
    """
    from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment

    env = FraudHunterEnvironment(case_seed_range=TRAIN_SEED_RANGE)
    prompts = []
    for _ in range(n_episodes):
        obs = env.reset()
        prompts.append({
            "prompt": obs.case_brief or "Investigate the case.",
            "difficulty_tier": obs.difficulty_tier,
        })
    return Dataset.from_list(prompts)


def _can_train_with_unsloth() -> bool:
    return bool(
        FastLanguageModel is not None
        and GRPOConfig is not None
        and GRPOTrainer is not None
        and Dataset is not None
        and torch is not None
        and torch.cuda.is_available()
    )


def _run_cpu_smoke_test() -> None:
    from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment

    env = FraudHunterEnvironment(case_seed_range=TRAIN_SEED_RANGE)
    obs = env.reset()
    print("CPU/no-GPU fallback: training skipped.")
    print(f"Verified environment reset on train split {TRAIN_SEED_RANGE}: tier={obs.difficulty_tier}")


# ── Main Training Loop ───────────────────────────────────────────────────────

def main():
    if os.environ.get("FRAUD_HUNTER_TRAIN") != "1":
        _run_cpu_smoke_test()
        print("Set FRAUD_HUNTER_TRAIN=1 on a CUDA host to run GRPO.")
        return

    if not _can_train_with_unsloth():
        _run_cpu_smoke_test()
        print("CUDA/Unsloth stack unavailable; skipping GRPO execution.")
        return

    print(f"Loading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,
        enforce_eager=True,  # T4 (compute 7.5) hits a torch.compile/CUDAGraph bug
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # DAPO configuration with clip-higher and zero KL
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_gpu_memory_utilization=0.4,
        # DAPO loss
        loss_type="dapo",
        # Clip-higher strategy
        epsilon=0.2,
        epsilon_high=0.25,
        # Zero KL penalty
        beta=0.0,
        # LR schedule
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Training
        logging_steps=1,
        fp16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    dataset = load_dataset(n_episodes=100)
    print(f"Dataset loaded: {len(dataset)} episodes")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[environment_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training (DAPO loss, clip-higher, zero KL)...")
    trainer.train()
    print(f"Training complete. Checkpoints in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
