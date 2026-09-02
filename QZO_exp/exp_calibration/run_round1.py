# SPDX-License-Identifier: MIT
"""Full round-1 fine-tune (2700 steps = 200 epochs x 54 windows / n_accum 4 — the exp18/exp5
recipe) under the NEW calibration (pooled@99.99), host PyTorch+Brevitas simulation with the
device update rule. Extends results.json under keys '<cfg>|<mode>|round1'.

Runs direct (the requested number) plus master as the control, both calibrations for context.
Same shared z / window schedule / seed as the 300-step regimes (steps 0..299 identical).
"""
import time

import numpy as np

import run_study as S

ROUND1_STEPS = 200 * S.N_TRAIN // S.N_ACCUM      # 2700

def main():
    r = S.load_results()
    regs = r.setdefault("regimes", {})
    _, trX, trY, evX, evY = S.get_data()
    for key in (S.best_pooled_key(r), "old54"):
        for mode in ("direct", "master"):
            tag = f"{key}|{mode}|round1"
            if tag in regs:
                S.log(f"[round1] {tag} done, skipping")
                continue
            _, model = S.build_model()
            S.freeze_config(model, r, key, trX)
            params = S.build_params(model)
            S.log(f"[round1] {tag} lr={S.LR_FT:g} {ROUND1_STEPS} steps")
            regs[tag] = S.run_regime(model, params, trX, trY, evX, evY,
                                     mode, S.LR_FT, ROUND1_STEPS, tag)
            S.save_results(r)
    S.log("==== round1 complete ====")
    for tag, d in regs.items():
        if tag.endswith("|round1"):
            S.log(f"  {tag:32s} bal={d['balanced_accuracy']:.2f}%  "
                  f"conv-int8-changed={d['cum_pct_convw_int8_changed']:.2f}%  "
                  f"zero-move-steps={d['pct_steps_zero_conv_movement']:.1f}%")

if __name__ == "__main__":
    main()
