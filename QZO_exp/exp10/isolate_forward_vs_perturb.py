# SPDX-License-Identifier: MIT
"""exp10 — CORRECTED isolation: forward-fidelity vs perturbation-operator.

The exp5 matched-z test compared A (Brevitas) and B (integer graph) losses "with the same z"
— but A applies a UNIFORM per-channel step (dz_int[c] for every element) while B's kernel
computes per-element (z*mul + 2^14)>>15 with sign-dependent rounding. So the two were
evaluated at DIFFERENT perturbed states, conflating two effects. Here we separate them:

  1. Quantify the state difference: element-wise diff between A's own +eps state and B's
     +eps state (same z).
  2. Evaluate BOTH forwards at B's EXACT +eps/-eps state (conv codes copied verbatim;
     BN/fc float values copied; conv biases dequantized rqsadd -> real -> A's s_b grid,
     error ~s_b/2 ~ 1e-4, negligible):
         if losses now agree to the unperturbed level (~0.03) -> forwards are faithful and
         the exp5 discrepancy was the PERTURBATION OPERATOR;
         if they still differ by ~0.3-0.7 -> genuine loss-landscape divergence.

Run in agitated_hugle.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F

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
# per-layer conv OUTPUT scales (the Dequant scales; verified in the exported graph)
S_OUT = {0: 13.1875, 1: 0.2120361328125, 2: 0.08878418803215027,
         3: 0.12207034230232239, 4: 0.04204712435603142}


def b_perturbed(tag, sign):
    out = f"/tmp/_iso_{tag}.onnx"
    build_qzo_train_graph(str(FIX / "network.onnx"), out, eps=sign * EPS, seed=SEED)
    m = onnx.load(out)
    inp = np.load(FIX / "inputs.npz")
    names = [i.name for i in m.graph.input]
    feed = {n: inp[f"arr_{i:04d}"] for i, n in enumerate(names)}
    nodes = [n for n in m.graph.node if n.op_type in ("RQSPerturbRademacher", "PerturbRademacher")]
    v = run_onnx_graph(out, feed, output_names=[n.output[0] for n in nodes])
    return {n.input[0]: np.asarray(x) for n, x in zip(nodes, v)}, feed


def a_setup():
    S.SEED = SEED
    r = S.load_results()
    _, qm = L.build_qmodel(3)
    S.freeze_config(qm, r, "pooled@99.99", None)
    params = S.build_params(qm)
    params = RF.to_float_fc(qm, params)
    return qm, params


def a_loss_at_state(qm, params, state, X, Y, mbs):
    with torch.no_grad():
        S.install(params, state, "direct")
        return [float(F.cross_entropy(qm(torch.from_numpy(X[i:i + 1])),
                                      torch.from_numpy(Y[i:i + 1]), reduction="sum")) for i in mbs]


def map_b_state_into_a(P, params):
    """B's perturbed tensors -> A's state dict (codes for quant, values for float)."""
    pby = {p["name"]: p for p in params}
    st = {}
    for i in range(5):
        # conv weight codes: identical int8 domain, copy verbatim
        w = np.asarray(P[f"blocks.{i}.conv.weight_int8"])
        p = pby[f"blocks.{i}.conv.weight"]
        st[p["name"]] = torch.from_numpy(w.astype(np.float32).reshape(p["init"].shape))
        # conv bias: rqsadd -> real -> A's s_b grid
        add = np.asarray(P[f"blocks.{i}.conv.bias_rqsadd"]).astype(np.float64)
        b_real = add * S_OUT[i] / (1 << 16)
        pb = pby[f"blocks.{i}.conv.bias"]
        st[pb["name"]] = torch.from_numpy(
            np.round(b_real / pb["scale"].reshape(-1).numpy()).astype(np.float32))
        for part in ("weight", "bias"):
            bn = f"blocks.{i}.bn.{part}"
            st[bn] = torch.from_numpy(np.asarray(P[bn]).astype(np.float32).reshape(
                pby[bn]["init"].shape))
    for bk, ak in (("fc.weight_fp32", "fc.weight"), ("fc.bias_fp32", "fc.bias")):
        st[ak] = torch.from_numpy(np.asarray(P[bk]).astype(np.float32).reshape(
            pby[ak]["init"].shape))
    return st


def main():
    d = np.load("/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz")
    X, Y = d["trX1"], d["trY1"]
    mbs = [0, 1, 2, 3]
    qm, params = a_setup()

    # ---- 1. element-wise state difference: A's own +eps state vs B's +eps state -----------
    P_plus, base = b_perturbed("p", +1)
    P_minus, _ = b_perturbed("m", -1)
    pby = {p["name"]: p for p in params}
    print("1) element-wise +eps state difference (A's uniform dz_int vs B's per-element):")
    tot = diff = 0
    for i in range(5):
        bk = f"blocks.{i}.conv.weight_int8"
        b0 = np.asarray(base[bk]).astype(np.int64).ravel()
        dpB = np.asarray(P_plus[bk]).astype(np.int64).ravel() - b0
        p = pby[f"blocks.{i}.conv.weight"]
        z = np.sign(dpB); z[z == 0] = 1                      # B's z (0-steps -> arbitrary +1)
        dz = p["dz_int"].reshape(-1).numpy()
        oc = p["init"].shape[0]
        dpA_unclamped = np.repeat(dz, dpB.size // oc) * z
        a_state = np.clip(b0 + dpA_unclamped, -127, 127)
        b_state = b0 + dpB
        n = b0.size; nd = int((a_state != b_state).sum())
        tot += n; diff += nd
        print(f"   blocks.{i}: {nd}/{n} elements differ ({100*nd/n:.1f}%)")
    print(f"   TOTAL: {diff}/{tot} = {100*diff/tot:.1f}% of conv weights at a different code")

    # ---- 2. both forwards at B's EXACT perturbed states -----------------------------------
    for tag, P in (("+eps", P_plus), ("-eps", P_minus)):
        stA = map_b_state_into_a(P, params)
        lA = a_loss_at_state(qm, params, stA, X, Y, mbs)
        print(f"\n2) state = B's {tag} perturbed weights, losses per mb:")
        print(f"   A(at B-state): {[round(v,4) for v in lA]}")
    mz = json.load(open("/app/Onnx4Deeploy/QZO_exp/exp5_A_vs_B/matched_z_losses.json"))
    print(f"   B(itself) L+ : {[round(p,4) for p,_ in mz['b_pairs']]}")
    print(f"   B(itself) L- : {[round(m,4) for _,m in mz['b_pairs']]}")
    print(f"   (A at A-state, exp5): L+ {[round(p,4) for p,_ in mz['a_pairs']]}")
    print(f"                         L- {[round(m,4) for _,m in mz['a_pairs']]}")


if __name__ == "__main__":
    main()
