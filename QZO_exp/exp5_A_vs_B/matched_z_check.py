# SPDX-License-Identifier: MIT
"""Step 1 — matched-z cross-validation of the INTEGER training path (B) against Brevitas (A).

Why: A (Brevitas fake-quant) and B (exported integer graph) agree on the FORWARD (zero-shot
85.00 = 85.00, cos 0.9994) but train to opposite outcomes (+3.89 vs -1.11 on round 1). Their
z streams differ (numpy RandomState vs the device xorshift32), so trajectory comparison is
confounded. Here we remove the confound: extract B's ACTUAL z per parameter, feed the SAME z
into A, and compare — from identical weights — the quantities that drive training:

  1. the perturbation each path applies, in LSB units  (tests RQSPerturb magnitude/direction)
  2. L+, L-, and g = (L+-L-)/(2*eps*n_accum)           (tests the ZO signal itself)

B's z is recovered analytically: RQSPerturb computes noise = (z*mul + rounding) >> S, so
z = sign of (perturbed - original). Deterministic given (seed, node_id).

Run in agitated_hugle.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

sys.path.insert(0, "/app/Onnx4Deeploy")
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/pytorch_ref")

import run_study as S            # noqa: E402
import stability_lib as L        # noqa: E402
import run_fc_float_ref as RF    # noqa: E402
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph  # noqa: E402
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph  # noqa: E402

FIX = Path("/app/Onnx4Deeploy/QZO_exp/exp9_full")
HERE = Path(__file__).resolve().parent
EPS, N_ACCUM, SEED = 0.01, 4, 42


def b_perturbed_weights(eps_sign):
    """Build B's train graph at +/-eps (seed 42, step 0) and return {input_name: perturbed value}
    by running ONLY the perturb nodes."""
    out = f"/tmp/_ab_{'p' if eps_sign > 0 else 'm'}.onnx"
    build_qzo_train_graph(str(FIX / "network.onnx"), out, eps=eps_sign * EPS, seed=SEED)
    m = onnx.load(out)
    inp = np.load(FIX / "inputs.npz")
    names = [i.name for i in m.graph.input]
    feed = {n: inp[f"arr_{i:04d}"] for i, n in enumerate(names)}
    pert_outs, pert_of = [], {}
    for n in m.graph.node:
        if n.op_type in ("RQSPerturbRademacher", "PerturbRademacher"):
            pert_outs.append(n.output[0]); pert_of[n.output[0]] = n.input[0]
    vals = run_onnx_graph(out, feed, output_names=pert_outs)
    return ({pert_of[o]: np.asarray(v) for o, v in zip(pert_outs, vals)},
            {n: feed[n] for n in names})


def main():
    S.SEED = SEED
    r = S.load_results()
    S.log_path = None
    # ---- 1. recover B's perturbation and z ------------------------------------------------
    P_plus, base = b_perturbed_weights(+1)
    P_minus, _ = b_perturbed_weights(-1)
    S_ = lambda x: np.asarray(x).reshape(-1)
    rep = {}
    print(f"{'param':30s} {'kind':6s} {'|dW| LSB (B)':>14s} {'z=+1 frac':>10s} {'antithetic?':>12s}")
    for k in P_plus:
        b0 = S_(base[k]).astype(np.float64)
        dp = S_(P_plus[k]).astype(np.float64) - b0
        dm = S_(P_minus[k]).astype(np.float64) - b0
        anti = bool(np.allclose(dp, -dm))
        kind = "int" if np.asarray(base[k]).dtype.kind in "iu" else "float"
        rep[k] = dict(kind=kind, mean_abs_step=float(np.abs(dp).mean()),
                      max_abs_step=float(np.abs(dp).max()), antithetic=anti,
                      frac_pos=float((dp > 0).mean()), n_zero=int((dp == 0).sum()), n=int(dp.size))
        print(f"{k:30s} {kind:6s} {np.abs(dp).mean():14.4f} {(dp>0).mean():10.3f} {str(anti):>12s}"
              + ("   ZERO-STEP!" if (dp == 0).any() else ""))
    json.dump(rep, open(HERE / "b_perturbation.json", "w"), indent=1)

    # ---- 2. B's losses at step 0 (from the export log, already computed) -------------------
    import re
    lg = open("/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/logs/export_full.log").read()
    b_pairs = [(float(a), float(b)) for a, b in
               re.findall(r"u0 a(?:\d) mb\d+: L\+=([0-9.]+) L-=([0-9.]+)", lg)][:N_ACCUM]
    b_g = sum(p - m for p, m in b_pairs) / (2 * EPS * N_ACCUM)
    print(f"\nB step-0 pairs: {[(round(p,4), round(m,4)) for p, m in b_pairs]}")
    print(f"B step-0 g_proj = {b_g:.4f}")

    # ---- 3. A with the SAME z ---------------------------------------------------------------
    data = L.load_fold_data(3)
    _, qm = L.build_qmodel(3)
    S.freeze_config(qm, r, "pooled@99.99", None)
    params = S.build_params(qm)
    params = RF.to_float_fc(qm, params)          # device-faithful fc-float
    name_map = {"blocks.%d.conv.weight_int8" % i: "blocks.%d.conv.weight" % i for i in range(5)}
    name_map.update({"blocks.%d.conv.bias_rqsadd" % i: "blocks.%d.conv.bias" % i for i in range(5)})
    name_map.update({"fc.weight_fp32": "fc.weight", "fc.bias_fp32": "fc.bias"})
    for i in range(5):
        name_map["blocks.%d.bn.weight" % i] = "blocks.%d.bn.weight" % i
        name_map["blocks.%d.bn.bias" % i] = "blocks.%d.bn.bias" % i
    pby = {p["name"]: p for p in params}
    zB = {}
    for bk, dp in ((k, S_(P_plus[k]).astype(np.float64) - S_(base[k]).astype(np.float64))
                   for k in P_plus):
        ak = name_map.get(bk)
        if ak in pby:
            zB[ak] = torch.from_numpy(np.sign(dp).reshape(pby[ak]["init"].shape).astype(np.float32))
    missing = [p["name"] for p in params if p["name"] not in zB]
    print(f"\nmapped z for {len(zB)}/{len(params)} params" + (f"  MISSING {missing}" if missing else ""))

    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    Xt, Yt = torch.from_numpy(data["trX1"]), torch.from_numpy(data["trY1"])
    a_pairs = []
    for a in range(N_ACCUM):
        xb, yb = Xt[a:a + 1], Yt[a:a + 1]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                dz = p["dz_int"] * zB[n]; di_p[n], di_m[n] = dz, -dz
            else:
                dz = EPS * zB[n]; df_p[n], df_m[n] = dz, -dz
        with torch.no_grad():
            S.install(params, state, "direct", di_p, df_p)
            Lp = float(F.cross_entropy(qm(xb), yb, reduction="sum"))
            S.install(params, state, "direct", di_m, df_m)
            Lm = float(F.cross_entropy(qm(xb), yb, reduction="sum"))
        a_pairs.append((Lp, Lm))
    a_g = sum(p - m for p, m in a_pairs) / (2 * EPS * N_ACCUM)
    print(f"A step-0 pairs (same z): {[(round(p,4), round(m,4)) for p, m in a_pairs]}")
    print(f"A step-0 g_proj = {a_g:.4f}")
    print(f"\n==> g_proj  A={a_g:.4f}  B={b_g:.4f}   ratio A/B = {a_g/b_g if b_g else float('nan'):.3f}")
    json.dump(dict(a_pairs=a_pairs, b_pairs=b_pairs, a_g=a_g, b_g=b_g),
              open(HERE / "matched_z_losses.json", "w"), indent=1)


if __name__ == "__main__":
    main()
