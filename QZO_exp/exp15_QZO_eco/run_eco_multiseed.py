# SPDX-License-Identifier: MIT
"""exp15 multiseed — fair ECO-vs-direct comparison across the SAME seeds/FT-draws as the
exp_calibration incremental_multiseed direct@1e-5 run.

Mirrors run_incremental_multiseed's seed coupling EXACTLY: the seed drives both the
Rademacher z-stream (draw_z uses S.SEED+u) AND the 54-window stratified FT draw, so each
seed has its own FT windows via the per-seed cache data_cache_incr_seed{seed}.npz. We reuse
those caches so ECO sees the identical FT data the direct@1e-5 multiseed saw at each seed.

seed 42 is already recorded in exp15 results.json (default cache). Here we add {1,7,123} and
pool to report mean +- std for ECO+SR, ECO memory-free RTN, and a direct@1e-5 sanity arm.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import run_eco as E        # noqa: E402  (sets up sys.path to exp_calibration)
import run_study as S      # noqa: E402
import run_incremental as RI  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = [1, 7, 123]
ARMS = ["eco_mf_3e6_b90_sr", "eco_mf_3e6_b90", "direct_1e5"]
ROUNDS = [1, 2, 3, 4]
STEPS = int(os.environ.get("STEPS", RI.STEPS))


def main():
    r = E.load_results()
    ms = r.setdefault("multiseed", {})
    base_seed = S.SEED
    base_cache = RI.CACHE
    for seed in SEEDS:
        skey = f"seed{seed}"
        ms.setdefault(skey, {})
        S.SEED = seed
        RI.CACHE = S.HERE / f"data_cache_incr_seed{seed}.npz"
        assert RI.CACHE.exists(), f"missing {RI.CACHE}"
        for arm in ARMS:
            if arm in ms[skey] and all(f"round{x}" in ms[skey][arm] for x in ROUNDS):
                E.log(f"[ms] seed{seed} {arm} done, skipping")
                continue
            cfg = dict(E.ARMS[arm]); cfg["_tag"] = f"s{seed}:{arm}"
            t0 = time.time()
            E.log(f"[ms] === seed {seed}: {arm} ===")
            ms[skey][arm] = E.eco_incremental(cfg, ms[skey].get(arm, {}), ROUNDS, STEPS)
            E.save_results(r)
            E.log(f"[ms] seed{seed} {arm} ({time.time()-t0:.0f}s)")
        S.SEED = base_seed
        RI.CACHE = base_cache

    # pool seed42 (from exp15 arms) + {1,7,123}
    seeds_used = [42] + SEEDS
    E.log(f"==== multiseed pooled over {seeds_used} ====")
    summary = {}
    for arm in ARMS:
        per_seed_mean = []
        for seed in seeds_used:
            src = r["arms"][arm] if seed == 42 else ms[f"seed{seed}"][arm]
            m = float(np.mean([src[f"round{i}"]["acc_after"] for i in ROUNDS]))
            per_seed_mean.append(m)
        arr = np.array(per_seed_mean)
        summary[arm] = dict(per_seed=dict(zip(map(str, seeds_used), per_seed_mean)),
                            mean=float(arr.mean()), std=float(arr.std(ddof=1)))
        E.log(f"  {arm:20s} mean4rd per seed {seeds_used} = "
              f"{[round(x,2) for x in per_seed_mean]}  ->  "
              f"{arr.mean():.2f} +- {arr.std(ddof=1):.2f}")
    r["multiseed_summary"] = dict(seeds=seeds_used, arms=summary)
    E.save_results(r)
    E.log("==== done ====")


if __name__ == "__main__":
    main()
