# SPDX-License-Identifier: MIT
"""exp4_ceiling — abs-ceiling update instead of rounding (supervisor's proposal).

Update rule under test, for the int-grid params (conv int8 weights + int32 biases):
    u     = coeff * z / s_w[c]
    delta = sign(u) * ceil(|u|)          # instead of round(u): ANY nonzero u moves >= 1 LSB
Float params (BN gamma/beta) update normally. Everything else = the standard round-1 protocol
at the FLOAT-ZO lr (3e-6): fold 3, 54 stratified seed-42 windows from sess-3 batch 1, 2700
steps (200 epochs, n_accum 4), eps 0.01, act scales pooled@99.99 + weight scales abs-max
frozen (fc int8, same as the run_incremental rows this compares against).

Measured: zero-shot + final b2 balanced accuracy, per-step conv-weight movement counts,
cumulative net + union movement. Baselines for comparison (same harness): round@3e-6 = 87.78%
(0% moved), round@1e-5 = 90.00% (65.9% net), zero-shot 85.56%.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
import run_study as S  # noqa: E402

HERE = Path(__file__).resolve().parent
S._LOG = open(HERE / "run.log", "a")
LR = 3e-6
STEPS = 200 * S.N_TRAIN // S.N_ACCUM


def main():
    S.SEED = 42
    r = S.load_results()
    d = np.load("/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz")
    trX, trY = torch.from_numpy(d["trX1"]), torch.from_numpy(d["trY1"])
    evX, evY = d["evX1"], d["evY1"]
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    init_int = {p["name"]: state[p["name"]].clone() for p in conv_w}
    ever = {p["name"]: torch.zeros_like(state[p["name"]], dtype=torch.bool) for p in conv_w}

    with torch.no_grad():
        S.install(params, state, "direct")
        zs, _, _ = S.balanced(S.logits_of(model, evX), evY)
    S.log(f"[ceiling] zero-shot b2 = {zs:.2f}%   lr={LR:g}  steps={STEPS}")

    moving_steps = 0
    moved_per_step = []
    t0 = time.time()
    for u in range(STEPS):
        z = S.draw_z(params, u)
        idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
        xb, yb = trX[idx], trY[idx]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                dz = p["dz_int"] * z[n]; di_p[n], di_m[n] = dz, -dz
            else:
                dz = S.EPS * z[n]; df_p[n], df_m[n] = dz, -dz
        with torch.no_grad():
            S.install(params, state, "direct", di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            S.install(params, state, "direct", di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
        coeff = -LR * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
        step_moved = 0
        for p in params:
            n = p["name"]
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            else:
                uu = coeff * z[n] / p["scale"]
                delta = torch.sign(uu) * torch.ceil(torch.abs(uu))   # ABS-CEILING rule
                if n.endswith(".conv.weight"):
                    ever[n] |= (delta != 0)
                    step_moved += int((delta != 0).sum())
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
        moved_per_step.append(step_moved)
        if step_moved:
            moving_steps += 1
        if (u + 1) % 675 == 0:
            with torch.no_grad():
                S.install(params, state, "direct")
                b, _, _ = S.balanced(S.logits_of(model, evX), evY)
            S.log(f"  [ceiling] step {u+1}/{STEPS} b2={b:.2f}% "
                  f"moved(this step)={100.0*step_moved/n_convw:.1f}% ({time.time()-t0:.0f}s)")
    with torch.no_grad():
        S.install(params, state, "direct")
        bal, ov, _ = S.balanced(S.logits_of(model, evX), evY)
    net = sum(int((state[p["name"]] != init_int[p["name"]]).sum()) for p in conv_w)
    uni = sum(int(e.sum()) for e in ever.values())
    res = dict(lr=LR, steps=STEPS, zero_shot=zs, final_b2=bal, overall=ov,
               moving_steps=moving_steps, pct_moving_steps=100.0 * moving_steps / STEPS,
               mean_moved_per_step_pct=100.0 * float(np.mean(moved_per_step)) / n_convw,
               net_changed_pct=100.0 * net / n_convw, union_ever_pct=100.0 * uni / n_convw)
    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    S.log(f"[ceiling] FINAL b2={bal:.2f}% (zero-shot {zs:.2f})  "
          f"moving-steps={moving_steps}/{STEPS}  mean-moved/step={res['mean_moved_per_step_pct']:.1f}%  "
          f"net={res['net_changed_pct']:.1f}%  union={res['union_ever_pct']:.1f}%")
    S.log("==== ceiling complete ====")


if __name__ == "__main__":
    main()
