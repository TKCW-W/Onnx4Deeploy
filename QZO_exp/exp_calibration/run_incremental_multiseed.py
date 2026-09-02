# SPDX-License-Identifier: MIT
"""Multi-seed confirmation of the incremental QZO-vs-float-ZO result. Repeats the 4-round
streaming protocol (run_incremental) at several seeds and reports per-batch mean +- std, to
(a) check my float ZO regresses toward the exp18 reference (b2-5 = 87.2 +- 0.3 multi-seed),
and (b) confirm QZO tracks float ZO across seeds.

The seed drives BOTH the Rademacher z-stream (draw_z / fp_draw_z use SEED+u) AND the 54-window
stratified FT draw (load_batches(seed=SEED)) — same coupling as run_incremental. The already-
recorded run_incremental result IS the seed-42 sample; we add new seeds and pool.
"""
import time

import numpy as np

import run_study as S
import run_incremental as RI

SEEDS = [1, 7, 123]


def main():
    r = S.load_results()
    ms = r.setdefault("incremental_multiseed", {})
    base_seed = S.SEED
    for seed in SEEDS:
        skey = f"seed{seed}"
        if skey in ms and "qzo" in ms[skey] and "fp" in ms[skey]:
            S.log(f"[ms] {skey} done, skipping")
            continue
        S.SEED = seed
        try:
            # per-seed FT/eval fixtures (eval batch is seed-independent; train draw is seeded)
            RI.CACHE = S.HERE / f"data_cache_incr_seed{seed}.npz"
            RI.stage_incr_data()
            S.log(f"[ms] === seed {seed}: QZO ===")
            t0 = time.time()
            q = RI.qzo_incremental({})
            S.log(f"[ms] === seed {seed}: float ZO === ({time.time()-t0:.0f}s for QZO)")
            f = RI.fp_incremental({})
            ms[skey] = dict(qzo=q, fp=f)
            r["incremental_multiseed"] = ms
            S.save_results(r)
        finally:
            S.SEED = base_seed
            RI.CACHE = S.HERE / "data_cache_incr.npz"

    # ---- pool seed 42 (from run_incremental) + new seeds ----
    S.log("==== multiseed complete — per-batch mean +- std (ddof=1) ====")
    inc42 = r.get("incremental", {})
    rows = {"qzo": {}, "fp": {}}
    for arm in ("qzo", "fp"):
        for rd in range(1, 5):
            eb = rd + 1
            vals = []
            if arm in inc42:
                vals.append(inc42[arm][f"round{rd}"]["acc_after"])
            for seed in SEEDS:
                vals.append(ms[f"seed{seed}"][arm][f"round{rd}"]["acc_after"])
            rows[arm][eb] = vals
    seeds_used = [42] + SEEDS
    S.log(f"  seeds pooled: {seeds_used}")
    for arm, label in (("fp", "float ZO"), ("qzo", "QZO int8@1e-5")):
        line = f"  {label:16s}"
        means = []
        for eb in (2, 3, 4, 5):
            v = np.array(rows[arm][eb])
            means.append(v.mean())
            line += f"  b{eb}={v.mean():5.2f}±{v.std(ddof=1):4.2f}"
        line += f"   b2-5={np.mean(means):5.2f}"
        S.log(line)
    # per-seed b2-5 means for the record
    for arm, label in (("fp", "float ZO"), ("qzo", "QZO")):
        for si, seed in enumerate(seeds_used):
            m = np.mean([rows[arm][eb][si] for eb in (2, 3, 4, 5)])
            S.log(f"    {label:8s} seed {seed:3d}: b2-5={m:.2f}")
    r["incremental_multiseed_summary"] = {
        arm: {f"b{eb}": rows[arm][eb] for eb in (2, 3, 4, 5)} for arm in ("qzo", "fp")}
    r["incremental_multiseed_summary"]["seeds"] = seeds_used
    S.save_results(r)


if __name__ == "__main__":
    main()
