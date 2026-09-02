# SPDX-License-Identifier: MIT
"""Ablation for the lr=1e-5 direct-int8 result (90.00%): freeze ALL int8 weights (conv+fc)
explicitly and train only the remaining parameters (BN gamma/beta in fp32, conv/fc biases on
their fine int32 grid) at the same lr, same z/windows/seed. If accuracy stays ~90%, the int8
weight movement contributes nothing and the gain is BN+bias learning; if it drops toward the
zero-shot, the weight movement matters.
Also runs the complementary ablation: ONLY int8 weights train (BN/bias frozen)."""
import numpy as np
import torch
import torch.nn.functional as F

import run_study as S

ROUND1_STEPS = 200 * S.N_TRAIN // S.N_ACCUM
LR = 1e-5

def run(model, params, trX, trY, evX, evY, update_weights: bool, update_rest: bool, tag: str):
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant"
                         else p["init"].clone()) for p in params}
    Xt, Yt = torch.from_numpy(trX), torch.from_numpy(trY)
    is_w = lambda p: p["kind"] == "quant" and p["name"].endswith(".weight")
    for u in range(ROUND1_STEPS):
        z = S.draw_z(params, u)
        idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
        xb, yb = Xt[idx], Yt[idx]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                d = p["dz_int"] * z[n]
                di_p[n], di_m[n] = d, -d
            else:
                d = S.EPS * z[n]
                df_p[n], df_m[n] = d, -d
        with torch.no_grad():
            S.install(params, state, "direct", di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            S.install(params, state, "direct", di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
        coeff = -LR * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
        for p in params:
            n = p["name"]
            allowed = update_weights if is_w(p) else update_rest
            if not allowed:
                continue
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            else:
                delta = torch.round(coeff * z[n] / p["scale"])
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
    with torch.no_grad():
        S.install(params, state, "direct")
        bal, ov, _ = S.balanced(S.logits_of(model, evX), evY)
    moved = sum(int((state[p["name"]] != S.q_int(p["init"], p)).sum())
                for p in params if is_w(p))
    tot = sum(int(p["init"].numel()) for p in params if is_w(p))
    S.log(f"[ablate] {tag:34s} bal={bal:6.2f}%  int8-w changed={100.0*moved/tot:6.2f}%")
    return dict(balanced_accuracy=bal, int8_w_changed_pct=100.0 * moved / tot)

def main():
    r = S.load_results()
    ab = r.setdefault("ablation_lr1e-05", {})
    _, trX, trY, evX, evY = S.get_data()
    for tag, uw, ur in [("frozen-int8-w (BN+bias only)", False, True),
                        ("only-int8-w (BN+bias frozen)", True, False)]:
        if tag in ab:
            S.log(f"[ablate] {tag} done, skipping")
            continue
        _, model = S.build_model()
        S.freeze_config(model, r, "pooled@99.99", trX)
        params = S.build_params(model)
        ab[tag] = run(model, params, trX, trY, evX, evY, uw, ur, tag)
        S.save_results(r)
    S.log("==== ablate complete ====")

if __name__ == "__main__":
    main()
