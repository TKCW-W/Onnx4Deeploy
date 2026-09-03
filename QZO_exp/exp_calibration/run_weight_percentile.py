# SPDX-License-Identifier: MIT
"""Does an abspercentile weight scale (instead of per-channel abs-max) help?

Mechanism: s_w[c] = percentile_p(|W[c]|)/127. Lower p -> smaller s_w -> finer grid -> lower
stall threshold |g|>=0.5*s_w/lr (more conv movement) BUT saturates the largest weights
(round(W/s_w) clamps at +-127). This measures which effect wins.

Per weight-percentile p in {100(abs-max baseline), 99.9, 99, 95}: zero-shot b2, then round-1
QZO (ft b1 -> eval b2, 2700 steps, seed 42) at lr 3e-6 (stalled) and 1e-5. Activation scales
frozen at pooled@99.99 throughout. Reports acc, conv-weights-moved, and %weights clipped.
"""
import numpy as np
import torch

import run_study as S

PCTS = [100.0, 99.9, 99.0, 95.0]
LRS = [3e-6, 1e-5]
STEPS = 200 * S.N_TRAIN // S.N_ACCUM


def set_weight_percentile(params, pct):
    """Override conv+fc weight scales (and dependent bias scales) to percentile-p. Returns
    {tensor: %clipped}."""
    byname = {p["name"]: p for p in params}
    clip = {}
    for p in params:
        if p["kind"] == "quant" and (p["name"].endswith(".conv.weight") or p["name"] == "fc.weight"):
            W = p["init"]; oc = W.shape[0]
            Wf = W.reshape(oc, -1).abs()
            if pct >= 100.0:
                thr = Wf.max(dim=1).values
            else:
                thr = torch.quantile(Wf, pct / 100.0, dim=1)
            thr = torch.clamp(thr, min=1e-12)
            new_scale = (thr / 127.0).reshape(p["scale"].shape).to(p["scale"].dtype)
            ratio = (new_scale.reshape(-1) / p["scale"].reshape(-1))
            clip[p["name"]] = 100.0 * float((Wf > thr[:, None]).float().mean())
            p["scale"] = new_scale
            p["dz_int"] = torch.round(S.EPS / new_scale)
            p["mod"].weight_quant.tensor_quant.scaling_impl = S.ConstScale(new_scale)
            bname = p["name"].replace(".weight", ".bias")
            if bname in byname:                       # s_b = s_in*s_w -> scales with ratio
                bp = byname[bname]
                bp["scale"] = (bp["scale"].reshape(-1) * ratio).reshape(bp["scale"].shape)
                bp["dz_int"] = torch.round(S.EPS / bp["scale"])
    return clip


def build(pct, r):
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    clip = set_weight_percentile(params, pct)
    return model, params, clip


def zeroshot(model, params, evX, evY):
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    with torch.no_grad():
        S.install(params, state, "direct")
        b, _, _ = S.balanced(S.logits_of(model, evX), evY)
    return b


def main():
    r = S.load_results()
    d = np.load(S.HERE / "data_cache_incr.npz")
    trX, trY, evX, evY = d["trX1"], d["trY1"], d["evX1"], d["evY1"]
    out = r.setdefault("weight_percentile", {})
    S.log(f"==== weight-scale abspercentile sweep (ft b1 -> eval b2, seed 42, {STEPS} steps) ====")
    for pct in PCTS:
        key = f"p{pct:g}"
        if key in out:
            S.log(f"[wpct] {key} done, skipping"); continue
        model, params, clip = build(pct, r)
        zs = zeroshot(model, params, evX, evY)
        clip_conv = np.mean([v for k, v in clip.items() if "conv" in k])
        rec = dict(pct=pct, zero_shot=zs, clip_pct=clip,
                   clip_conv_mean=float(clip_conv), clip_fc=clip.get("fc.weight"))
        for lr in LRS:
            model, params, _ = build(pct, r)             # fresh
            res = S.run_regime(model, params, trX, trY, evX, evY, "direct", lr, STEPS,
                               f"wpct{pct:g}|lr{lr:g}")
            rec[f"lr{lr:g}"] = dict(bal=res["balanced_accuracy"],
                                    conv_moved=res["cum_pct_convw_int8_changed"],
                                    zero_move_steps=res["pct_steps_zero_conv_movement"])
        out[key] = rec
        r["weight_percentile"] = out
        S.save_results(r)
        S.log(f"[wpct] p={pct:<5g} zeroshot={zs:5.2f}  clip(conv={clip_conv:.1f}% fc={clip.get('fc.weight',0):.1f}%)  "
              f"| 3e-6: bal={rec['lr3e-06']['bal']:.2f} moved={rec['lr3e-06']['conv_moved']:.1f}%  "
              f"| 1e-5: bal={rec['lr1e-05']['bal']:.2f} moved={rec['lr1e-05']['conv_moved']:.1f}%")
    S.log("==== weight_percentile complete ====")
    S.log(f"  {'p':>6s} {'zeroshot':>9s} {'clipConv':>9s} {'3e-6 bal':>9s} {'3e-6 mv':>8s} {'1e-5 bal':>9s} {'1e-5 mv':>8s}")
    for pct in PCTS:
        rec = out[f"p{pct:g}"]
        S.log(f"  {pct:6g} {rec['zero_shot']:9.2f} {rec['clip_conv_mean']:8.1f}% "
              f"{rec['lr3e-06']['bal']:9.2f} {rec['lr3e-06']['conv_moved']:7.1f}% "
              f"{rec['lr1e-05']['bal']:9.2f} {rec['lr1e-05']['conv_moved']:7.1f}%")


if __name__ == "__main__":
    main()
