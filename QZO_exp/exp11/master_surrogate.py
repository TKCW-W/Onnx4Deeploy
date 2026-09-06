# SPDX-License-Identifier: MIT
"""exp11: does a MASTER-WEIGHT update keep two ~1e-7-seeded trajectories within float-ZO tolerance,
where the DIRECT-int8 update (measured device-vs-host) cascades? Faithful replica of
_export_qzo_training's multi-step loop; --mode direct|master; --inject 0|1 nudges every L+/L- by
~+-delta relative (deterministic sign) = the measured device-host step-0 seed magnitude."""
import argparse, os, struct, sys, time
import numpy as np, onnx
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph, _iter_qzo_params
from onnx4deeploy.transform.qzo_weight_integerize import build_int8_forward as _bif
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph, _perturb_rqs_rademacher, _perturb_rademacher

ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--inject", type=int, default=0)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--delta", type=float, default=3e-7); ap.add_argument("--out", required=True); ap.add_argument("--selfcheck", type=int, default=0)
A = ap.parse_args()
EXP = "/app/Onnx4Deeploy/QZO_exp"; NET = f"{EXP}/exp10/baked_200ep/network.onnx"
EPS, SEED, N_ACCUM = 0.01, 42, 4; PERT = ("RQSPerturbRademacher", "PerturbRademacher")
tag = f"{A.mode}_inj{A.inject}"; QP, QM = f"/tmp/_sur_{tag}_p.onnx", f"/tmp/_sur_{tag}_m.onnx"
_, param_inputs = build_qzo_train_graph(NET, QP, eps=+EPS, seed=SEED); build_qzo_train_graph(NET, QM, eps=-EPS, seed=SEED)
mm_model, mm_scale = _bif(onnx.load(NET)); pmeta = list(_iter_qzo_params(mm_model, mm_scale))
P = {k: np.asarray(v).copy() for k, v in param_inputs.items()}
d = np.load(f"{EXP}/exp_calibration/data_cache_incr.npz"); X, Y = d["trX1"], d["trY1"]; DS = len(X)
mp, mmp = onnx.load(QP), onnx.load(QM); go = [o.name for o in mp.graph.output]
def patch(model, sd):
    for n in model.graph.node:
        if n.op_type in PERT:
            for a in n.attribute:
                if a.name == "seed": a.i = int(sd)
def bits(f): return struct.unpack(">I", struct.pack(">f", np.float32(f)))[0]
def inject(L, u, a, s):
    if not A.inject: return np.float32(L)
    sg = 1.0 if ((u * 7919 + a * 104729 + s * 31) % 2) == 0 else -1.0
    return np.float32(np.float32(L) * np.float32(1.0 + sg * A.delta))
rqs = [p for p in pmeta if p["kind"] == "rqs"]
# masters are fp32 RELATIVE to the initial integer value (init 0): full resolution even for the int32
# bias at its 32768 offset; read-out = clip(P0 + rint(M)) with the KERNEL's per-param clip from nlev.
P0 = {p["name"]: P[p["name"]].astype(np.int64).copy() for p in rqs}
M = {p["name"]: np.zeros(P[p["name"]].shape, np.float32) for p in rqs} if A.mode == "master" else None
def clipb(p): return -(p["nlev"] // 2) + 1, p["nlev"] // 2 - 1
def readout():
    for p in rqs:
        nm = p["name"]; lo, hi = clipb(p); P[nm] = np.clip(P0[nm] + np.rint(M[nm]).astype(np.int64), lo, hi).astype(P[nm].dtype)
rec = {"lp": [], "lm": [], "g": [], "snap_u": [], "snap": [], "msnap": []}
def snapshot(u):
    rec["snap_u"].append(u); rec["snap"].append(np.concatenate([P[p["name"]].reshape(-1).astype(np.int8) for p in rqs]))
    if M is not None: rec["msnap"].append(np.concatenate([M[p["name"]].reshape(-1) for p in rqs]))
def save():
    np.savez(A.out, lp=np.array(rec["lp"], np.float32), lm=np.array(rec["lm"], np.float32), g=np.array(rec["g"], np.float32),
             snap_u=np.array(rec["snap_u"]), snap=np.array(rec["snap"]), msnap=np.array(rec["msnap"]) if rec["msnap"] else np.zeros(0))
t0 = time.time()
for u in range(A.steps):
    seed_eff = SEED + u
    patch(mp, seed_eff); onnx.save(mp, QP); patch(mmp, seed_eff); onnx.save(mmp, QM)
    if M is not None and u > 0: readout()
    if u % 25 == 0: snapshot(u)
    acc = np.float32(0.0); lps, lms = [], []
    for a in range(N_ACCUM):
        mb = (u * N_ACCUM + a) % DS
        feed = {"input": X[mb:mb + 1].astype(np.float32), "label": np.asarray(Y[mb]).reshape(1, 1).astype(np.int64), **P}
        Lp = inject(next(np.asarray(r).flatten()[0] for r in run_onnx_graph(QP, feed, output_names=go) if np.asarray(r).size == 1), u, a, +1)
        Lm = inject(next(np.asarray(r).flatten()[0] for r in run_onnx_graph(QM, feed, output_names=go) if np.asarray(r).size == 1), u, a, -1)
        lps.append(Lp); lms.append(Lm); acc = np.float32(acc + (Lp - Lm))
    rec["lp"].append(lps); rec["lm"].append(lms)
    denom = np.float32(np.float32(np.float32(2.0) * np.float32(EPS)) * np.float32(N_ACCUM))
    g = np.float32(acc / denom); coeff = np.float32(np.float32(-A.lr) * g); ratio = float(np.float32(coeff) / np.float32(EPS))
    rec["g"].append(g)
    if u == 0: print("step0 L+ bits:", [f"{bits(x):08x}" for x in lps], flush=True)
    for p in pmeta:
        nm = p["name"]
        if p["kind"] == "rqs":
            mul = np.round(EPS / p["scale"] * p["div"]).astype(np.int32)
            if A.mode == "direct":
                P[nm] = _perturb_rqs_rademacher(P[nm], mul, seed_eff, p["idx"], p["div"], p["nlev"], 1, sign=1, eps_ratio=ratio).reshape(P[nm].shape).astype(P[nm].dtype)
            else:
                z0 = np.zeros(P[nm].shape, P[nm].dtype)
                rad = _perturb_rqs_rademacher(z0, np.full(mul.shape, p["div"], np.int32), seed_eff, p["idx"], p["div"], p["nlev"], 1, sign=1, eps_ratio=1.0).astype(np.float32).reshape(-1)
                size, nout = rad.size, mul.size; mpe = np.repeat(mul.astype(np.float32).reshape(-1), size // nout)[:size]
                if A.selfcheck and u == 0:   # my mapping must reproduce the int8 kernel's rounded delta exactly
                    S = int(np.log2(p["div"])); rnd = np.int64(1 << (S - 1))
                    m_val = np.rint(mpe * np.float32(ratio)).astype(np.int64)
                    dq = (rad.astype(np.int64) * m_val + rnd) >> S
                    ref = _perturb_rqs_rademacher(P[nm], mul, seed_eff, p["idx"], p["div"], p["nlev"], 1, sign=1, eps_ratio=ratio).astype(np.int64).reshape(-1)
                    lo, hi = clipb(p); mine = np.clip(P[nm].astype(np.int64).reshape(-1) + dq, lo, hi)
                    assert np.array_equal(ref, mine), f"mapping mismatch {nm}: {(ref != mine).sum()}"
                    print(f"selfcheck OK {nm}: rad=+-1 {np.all(np.abs(rad)==1)}, delta matches kernel", flush=True)
                dm = rad * (mpe * np.float32(ratio)) / np.float32(p["div"])
                M[nm] = (M[nm].reshape(-1) + dm).reshape(M[nm].shape).astype(np.float32)
        else:
            P[nm] = _perturb_rademacher(P[nm].astype(np.float32), seed_eff, p["idx"], float(coeff), 1).reshape(P[nm].shape).astype(np.float32)
    if u % 10 == 9 or u == A.steps - 1:
        save(); print(f"u{u} g={float(g):+.4f} {(time.time()-t0)/(u+1):.1f}s/step", flush=True)
snapshot(A.steps); save(); print("DONE", flush=True)
