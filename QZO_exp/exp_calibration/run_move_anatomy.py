# SPDX-License-Identifier: MIT
"""Replay the lr=1e-5 direct-int8 round-1 run (deterministic: same seed/z/windows) and
record the per-step movement anatomy: for every step, |g|, #conv-int8 weights moved,
#channels cleared, and the running union of ever-moved weights. Answers: how many of the
2700 steps moved anything, and how much moved per moving step vs the 65.93% cumulative."""
import json

import numpy as np
import torch
import torch.nn.functional as F

import run_study as S

ROUND1_STEPS = 200 * S.N_TRAIN // S.N_ACCUM
LR = 1e-5

def main():
    r = S.load_results()
    _, trX, trY, evX, evY = S.get_data()
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", trX)
    params = S.build_params(model)
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)

    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant"
                         else p["init"].clone()) for p in params}
    ever = {p["name"]: torch.zeros_like(state[p["name"]], dtype=torch.bool) for p in conv_w}
    prev = {p["name"]: state[p["name"]].clone() for p in conv_w}
    Xt, Yt = torch.from_numpy(trX), torch.from_numpy(trY)

    anat = []
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
        g = (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
        coeff = -LR * g
        for p in params:
            n = p["name"]
            if p["kind"] == "float":
                state[n] = state[n] + coeff * z[n]
            else:
                delta = torch.round(coeff * z[n] / p["scale"])
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
        moved = 0
        ch_moved = 0
        for p in conv_w:
            diff = state[p["name"]] != prev[p["name"]]
            moved += int(diff.sum())
            ever[p["name"]] |= diff
            # per-output-channel view: scale is per-channel; count channels with any movement
            nch = p["scale"].numel()
            ch_moved += int(diff.reshape(nch, -1).any(1).sum())
            prev[p["name"]] = state[p["name"]].clone()
        if moved:
            anat.append(dict(step=u, g=g, moved=moved, pct=100.0 * moved / n_convw,
                             channels=ch_moved))
    ever_n = sum(int(e.sum()) for e in ever.values())
    net = sum(int((state[p["name"]] != S.q_int(p["init"], p)).sum()) for p in conv_w)

    S.log(f"[anatomy] lr={LR:g}: {len(anat)}/{ROUND1_STEPS} steps moved anything "
          f"({100.0*len(anat)/ROUND1_STEPS:.1f}%)")
    pcts = [a["pct"] for a in anat]
    S.log(f"[anatomy] per moving step: moved weights min={min(pcts):.1f}% "
          f"median={float(np.median(pcts)):.1f}% max={max(pcts):.1f}% of {n_convw}")
    S.log(f"[anatomy] |g| on moving steps: min={min(abs(a['g']) for a in anat):.1f} "
          f"median={float(np.median([abs(a['g']) for a in anat])):.1f} "
          f"max={max(abs(a['g']) for a in anat):.1f}   (vs mean|g|~14.5 overall)")
    S.log(f"[anatomy] ever-moved union: {100.0*ever_n/n_convw:.2f}%   "
          f"net changed vs init: {100.0*net/n_convw:.2f}%  (bounces cancel)")
    S.log(f"[anatomy] total conv channels = 104")
    for a in anat[:12]:
        S.log(f"   step {a['step']:4d}  g={a['g']:+8.1f}  moved {a['moved']:6d} "
              f"({a['pct']:5.1f}%)  channels {a['channels']:3d}/104")
    r.setdefault("anatomy", {})["lr1e-05"] = dict(
        n_moving_steps=len(anat), n_steps=ROUND1_STEPS,
        per_step=anat, ever_moved_pct=100.0 * ever_n / n_convw,
        net_changed_pct=100.0 * net / n_convw)
    S.save_results(r)
    S.log("==== anatomy complete ====")

if __name__ == "__main__":
    main()
