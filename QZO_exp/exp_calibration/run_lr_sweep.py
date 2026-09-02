# SPDX-License-Identifier: MIT
"""Direct-int8 lr sweep: can a higher lr overcome the LSB stall while preserving accuracy?

Stall criterion: |lr*g/s_w[c]| >= 0.5. With mean|g|~14.5 and s_w in [0.0018, 0.0036] the
predicted onset of conv int8 movement is lr ~ 6e-5..1.2e-4. Sweep brackets that from the
known-stalled 3e-6 up to the expected-unstable 3e-4. Direct mode only, pooled@99.99
calibration, full round-1 protocol (2700 steps = 200 epochs x 54 / n_accum 4).

Results extend results.json under 'pooled@99.99|direct|lr<lr>|round1'.
"""
import time

import numpy as np

import run_study as S

ROUND1_STEPS = 200 * S.N_TRAIN // S.N_ACCUM      # 2700
LRS = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4]

def main():
    r = S.load_results()
    regs = r.setdefault("regimes", {})
    _, trX, trY, evX, evY = S.get_data()
    key = "pooled@99.99"
    for lr in LRS:
        tag = f"{key}|direct|lr{lr:g}|round1"
        if tag in regs:
            S.log(f"[lrsweep] {tag} done, skipping")
            continue
        _, model = S.build_model()
        S.freeze_config(model, r, key, trX)
        params = S.build_params(model)
        S.log(f"[lrsweep] {tag} lr={lr:g} {ROUND1_STEPS} steps")
        regs[tag] = S.run_regime(model, params, trX, trY, evX, evY,
                                 "direct", lr, ROUND1_STEPS, tag)
        S.save_results(r)
    S.log("==== lrsweep complete ====")
    S.log(f"  {'run':44s} {'bal%':>7s} {'conv-chg%':>10s} {'zero-mv%':>9s}")
    base = regs.get("pooled@99.99|direct|round1")
    if base:
        S.log(f"  {'lr3e-06 (baseline)':44s} {base['balanced_accuracy']:7.2f} "
              f"{base['cum_pct_convw_int8_changed']:10.2f} "
              f"{base['pct_steps_zero_conv_movement']:9.1f}")
    for lr in LRS:
        d = regs[f"{key}|direct|lr{lr:g}|round1"]
        S.log(f"  {f'lr{lr:g}':44s} {d['balanced_accuracy']:7.2f} "
              f"{d['cum_pct_convw_int8_changed']:10.2f} "
              f"{d['pct_steps_zero_conv_movement']:9.1f}")

if __name__ == "__main__":
    main()
