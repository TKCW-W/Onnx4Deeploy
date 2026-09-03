# SPDX-License-Identifier: MIT
"""Incremental 4-round streaming protocol, direct int8, at lr 3e-6 (the STALLED setting) —
the control for the lr 1e-5 run. Same seeds (42/1/7/123), same fixtures, act+weight scales
frozen. Since conv int8 weights stall at 3e-6, this isolates the BN+bias-only contribution
across the incremental rounds. Stored under r['incremental_3e6'] (seed 42) and
r['incremental_multiseed_3e6'] (all seeds); pooled summary printed."""
import time

import numpy as np

import run_study as S
import run_incremental as RI

SEEDS = [42, 1, 7, 123]
LR = 3e-6


def main():
    r = S.load_results()
    store = r.setdefault("incremental_3e6_allseeds", {})
    base_seed, base_lr, base_cache = S.SEED, RI.LR_QZO, RI.CACHE
    RI.LR_QZO = LR
    try:
        for seed in SEEDS:
            skey = f"seed{seed}"
            if skey in store:
                S.log(f"[3e6] {skey} done, skipping")
                continue
            S.SEED = seed
            RI.CACHE = S.HERE / (f"data_cache_incr.npz" if seed == 42
                                 else f"data_cache_incr_seed{seed}.npz")
            RI.stage_incr_data()
            S.log(f"[3e6] === seed {seed}: QZO direct lr=3e-6 ===")
            t0 = time.time()
            store[skey] = RI.qzo_incremental({})
            r["incremental_3e6_allseeds"] = store
            S.save_results(r)
            S.log(f"[3e6] seed {seed} done ({time.time()-t0:.0f}s)")
    finally:
        S.SEED, RI.LR_QZO, RI.CACHE = base_seed, base_lr, base_cache

    S.log("==== incremental 3e-6 complete — per-batch mean ± std (ddof=1) ====")
    S.log(f"  seeds pooled: {SEEDS}")
    means = []
    line_acc = "  QZO int8@3e-6   "
    for rd in range(1, 5):
        eb = rd + 1
        v = np.array([store[f"seed{s}"][f"round{rd}"]["acc_after"] for s in SEEDS])
        mv = np.array([store[f"seed{s}"][f"round{rd}"]["convw_moved_this_round_pct"] for s in SEEDS])
        means.append(v.mean())
        line_acc += f"  b{eb}={v.mean():5.2f}±{v.std(ddof=1):4.2f}"
        S.log(f"    round{rd} b{eb}: acc {v.mean():.2f}±{v.std(ddof=1):.2f}  "
              f"conv-moved {mv.mean():.2f}±{mv.std(ddof=1):.2f}%")
    line_acc += f"   b2-5={np.mean(means):5.2f}"
    S.log(line_acc)
    for s in SEEDS:
        m = np.mean([store[f"seed{s}"][f"round{rd}"]["acc_after"] for rd in range(1, 5)])
        S.log(f"    QZO@3e-6 seed {s:3d}: b2-5={m:.2f}")
    S.save_results(r)


if __name__ == "__main__":
    main()
