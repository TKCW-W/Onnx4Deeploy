# SPDX-License-Identifier: MIT
"""Robustness check around the lr-sweep winner (direct int8, pooled@99.99, round-1 protocol):
- plateau shape: lr 6e-6 and 2e-5 at the base z-seed
- seed stability: lr 1e-5 with two well-separated z-seeds (12345, 67890; draw_z uses
  RandomState(SEED+u), so nearby seeds would give overlapping z sequences)
Windows/data are cached and fixed; only the Rademacher z sequence changes with SEED.
"""
import numpy as np

import run_study as S

ROUND1_STEPS = 200 * S.N_TRAIN // S.N_ACCUM      # 2700
RUNS = [(6e-6, 42), (2e-5, 42), (1e-5, 12345), (1e-5, 67890)]

def main():
    r = S.load_results()
    regs = r.setdefault("regimes", {})
    _, trX, trY, evX, evY = S.get_data()
    key = "pooled@99.99"
    base_seed = S.SEED
    for lr, seed in RUNS:
        tag = f"{key}|direct|lr{lr:g}|seed{seed}|round1"
        if tag in regs:
            S.log(f"[lrconfirm] {tag} done, skipping")
            continue
        S.SEED = seed
        try:
            _, model = S.build_model()
            S.freeze_config(model, r, key, trX)
            params = S.build_params(model)
            S.log(f"[lrconfirm] {tag} lr={lr:g} seed={seed} {ROUND1_STEPS} steps")
            regs[tag] = S.run_regime(model, params, trX, trY, evX, evY,
                                     "direct", lr, ROUND1_STEPS, tag)
            regs[tag]["z_seed"] = seed
            S.save_results(r)
        finally:
            S.SEED = base_seed
    S.log("==== lrconfirm complete ====")
    for lr, seed in RUNS:
        d = regs[f"{key}|direct|lr{lr:g}|seed{seed}|round1"]
        S.log(f"  lr{lr:g} seed{seed:6d}  bal={d['balanced_accuracy']:6.2f}%  "
              f"conv-chg={d['cum_pct_convw_int8_changed']:6.2f}%  "
              f"zero-mv={d['pct_steps_zero_conv_movement']:5.1f}%")

if __name__ == "__main__":
    main()
