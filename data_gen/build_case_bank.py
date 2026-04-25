"""
Case-bank builder — fully local, no external APIs.

Generates one directory per case using `generate_multimodal_aks_case`:

    data/case_bank/
        tier_1/
            t1_<8hex>/
                medicare_records.db
                intercepted_comms/
                scanned_claims/
        tier_2/...

Modes:
    --mode test         → 10 cases per tier (50 total). For local debugging.
    --mode production   → 2000 cases per tier (10,000 total). For GRPO training.
    --mode custom       → use --count explicitly.

Parallelism:
    Uses multiprocessing.Pool sized to --workers (default: os.cpu_count()).
    Each worker is given a deterministic seed derived from (tier, index).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from .case_compiler import generate_multimodal_aks_case


ROOT = Path(__file__).resolve().parents[1]
CASE_BANK_DIR = ROOT / "data" / "case_bank"


MODE_COUNTS = {
    "test": 10,
    "production": 2000,
}


def _build_one(args: Tuple[str, int, str, int]) -> str:
    """Worker: generate one case. Returns case directory path as string."""
    tier_dir_str, tier, case_id, seed = args
    tier_dir = Path(tier_dir_str)
    path = generate_multimodal_aks_case(tier_dir, case_id, tier=tier, rng_seed=seed)
    return str(path)


def build_tier(
    outdir: Path,
    tier: int,
    count: int,
    workers: int,
    base_seed: int,
) -> int:
    tier_dir = outdir / f"tier_{tier}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Tuple[str, int, str, int]] = []
    for i in range(count):
        case_id = f"t{tier}_{uuid.uuid4().hex[:8]}"
        seed = base_seed + tier * 1_000_000 + i
        jobs.append((str(tier_dir), tier, case_id, seed))

    t0 = time.perf_counter()
    done = 0
    print(f"[build] tier {tier}: {count} cases across {workers} workers...")

    if workers <= 1:
        for job in jobs:
            _build_one(job)
            done += 1
            if done % max(1, count // 10) == 0:
                print(f"  tier {tier}: {done}/{count}")
    else:
        with Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(_build_one, jobs, chunksize=4):
                done += 1
                if done % max(1, count // 10) == 0:
                    print(f"  tier {tier}: {done}/{count}")

    dt = time.perf_counter() - t0
    rate = count / dt if dt > 0 else 0
    print(f"[build] tier {tier} done in {dt:.1f}s ({rate:.1f} cases/s)")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["test", "production", "custom"],
                        default="test",
                        help="test=10/tier, production=2000/tier, custom=--count")
    parser.add_argument("--count", type=int, default=None,
                        help="Cases per tier (overrides --mode for custom)")
    parser.add_argument("--tier", type=int, default=None,
                        help="Only build this tier (1-5). Default: all tiers.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--outdir", type=str, default=None,
                        help=f"Output directory (default: {CASE_BANK_DIR})")
    parser.add_argument("--seed", type=int, default=20260424,
                        help="Base seed for reproducibility")
    args = parser.parse_args()

    if args.mode == "custom":
        if args.count is None:
            parser.error("--mode custom requires --count")
        count_per_tier = args.count
    else:
        count_per_tier = args.count if args.count is not None else MODE_COUNTS[args.mode]

    outdir = Path(args.outdir) if args.outdir else CASE_BANK_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    tiers = [args.tier] if args.tier else [1, 2, 3, 4, 5]
    total_target = count_per_tier * len(tiers)
    print(f"[config] mode={args.mode} tiers={tiers} count/tier={count_per_tier} "
          f"workers={args.workers} outdir={outdir}")
    print(f"[config] building {total_target} cases total")

    t0 = time.perf_counter()
    total_done = 0
    for tier in tiers:
        total_done += build_tier(outdir, tier, count_per_tier, args.workers, args.seed)

    dt = time.perf_counter() - t0
    print(f"[done] {total_done}/{total_target} cases in {dt:.1f}s "
          f"(~{total_done / dt if dt else 0:.1f} cases/s)")
    print(f"[done] case bank at {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
