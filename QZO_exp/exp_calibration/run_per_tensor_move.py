# SPDX-License-Identifier: MIT
"""Per-tensor movement breakdown at lr 1e-5 (round-1 replay, seed 42, deterministic). For
every trainable tensor reports: kind, #elements, #moved (net vs init), %moved, and — for the
quant tensors — the per-channel s_w range, since the stall threshold |g| >= 0.5*s_w/lr is
per-channel. Answers whether some tensors are entirely untouched."""
import numpy as np
import torch
import torch.nn.functional as F

import run_study as S

STEPS = 200 * S.N_TRAIN // S.N_ACCUM
LR = 1e-5


def main():
    r = S.load_results()
    d = np.load(S.HERE / "data_cache_incr.npz")
    trX, trY = torch.from_numpy(d["trX1"]), torch.from_numpy(d["trY1"])
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant"
                         else p["init"].clone()) for p in params}
    init_int = {p["name"]: S.q_int(p["init"], p) for p in params if p["kind"] == "quant"}

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
        for p in params:
            n = p["name"]
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            else:
                state[n] = torch.clamp(state[n] + torch.round(coeff * z[n] / p["scale"]),
                                       p["lo"], p["hi"])

    S.log(f"==== per-tensor movement at lr {LR:g}, round-1 seed 42 ({STEPS} steps) ====")
    S.log(f"  {'tensor':22s} {'kind':6s} {'#elem':>7s} {'#moved':>7s} {'%moved':>7s}  s_w/scale range")
    rec = {}
    for p in params:
        n = p["name"]
        if p["kind"] == "quant":
            fin = state[n]; moved = int((fin != init_int[n]).sum())
            sc = p["scale"].reshape(-1).numpy()
            sr = f"s_w [{sc.min():.5f},{sc.max():.5f}]" if n.endswith(".weight") \
                 else f"s_b [{sc.min():.2e},{sc.max():.2e}]"
        else:
            fin = state[n]; moved = int((fin != p["init"]).sum()); sr = "fp32 (BN)"
        ne = int(p["init"].numel())
        rec[n] = dict(kind=p["kind"], n=ne, moved=moved, pct=100.0 * moved / ne)
        S.log(f"  {n:22s} {p['kind']:6s} {ne:7d} {moved:7d} {100.0*moved/ne:6.1f}%  {sr}")
    # roll up conv weights only
    cw = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    tot = sum(rec[p["name"]]["n"] for p in cw); mv = sum(rec[p["name"]]["moved"] for p in cw)
    S.log(f"  --> conv WEIGHTS total: {mv}/{tot} = {100.0*mv/tot:.1f}% moved")
    untouched = [n for n, v in rec.items() if v["kind"] == "quant" and v["moved"] == 0]
    S.log(f"  --> fully-untouched quant tensors: {untouched if untouched else 'NONE'}")
    r["per_tensor_move_1e5"] = rec
    S.save_results(r)
    S.log("==== per-tensor complete ====")


if __name__ == "__main__":
    main()
