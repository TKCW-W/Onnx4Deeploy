# SPDX-License-Identifier: MIT
"""exp11 step-0 bit-exactness checker (host run_onnx_graph vs device logged lp_bits).

Faithful to _export_qzo_training's loop at u=0: +eps graph built at seed=42, eps=+0.01,
initial param_inputs, windows mb=0..3, loss = the size-1 output (SCE `loss`, same quantity
the device logs as lp_bits). Run inside agitated_hugle:
  python3 /app/Onnx4Deeploy/QZO_exp/exp11/check_step0.py
"""
import struct, sys
import numpy as np, onnx
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

EXP = "/app/Onnx4Deeploy/QZO_exp"
FIX = f"{EXP}/exp12/baked_3e6"
EPS, SEED, N_ACCUM = 0.01, 42, 4

dev_hex = [l.strip() for l in open(f"{EXP}/exp11/device_lp_bits.txt") if l.strip()]
dev = np.array([struct.unpack(">f", bytes.fromhex(h))[0] for h in dev_hex], np.float32)

d = np.load(f"{EXP}/exp_calibration/data_cache_incr.npz")
X, Y = d["trX1"], d["trY1"]
print(f"windows: X{X.shape} {X.dtype}  Y{Y.shape}  (data_size={len(X)})")

QP = "/tmp/_exp11_qp.onnx"
_, P = build_qzo_train_graph(f"{FIX}/network.onnx", QP, eps=+EPS, seed=SEED)
from onnx4deeploy.transform.qzo_transform import freeze_conv_pmul
_m = onnx.load(QP); print("frozen pmul initializers:", freeze_conv_pmul(_m)); onnx.save(_m, QP)
# sanity: initial params == the fixture the device was built from (inputs.npz arr_ by input order)
m = onnx.load(QP); names = [i.name for i in m.graph.input]
inp = np.load(f"{FIX}/inputs.npz")
bad = [n for j, n in enumerate(names) if n in P and not np.array_equal(np.asarray(P[n]), inp[f"arr_{j:04d}"])]
print("param_inputs == fixture inputs.npz:", "OK" if not bad else f"MISMATCH {bad[:3]}")
go = [o.name for o in m.graph.output]

def bits(f): return struct.unpack(">I", struct.pack(">f", np.float32(f)))[0]

print(f"\n{'a':>2} {'host_loss':>12} {'host_bits':>10} {'dev_bits':>10} {'exact':>5} {'rel_resid':>10}")
n_exact = 0
for a in range(N_ACCUM):
    feed = {"input": X[a:a+1].astype(np.float32), "label": np.asarray(Y[a]).reshape(1, 1).astype(np.int64), **P}
    r = run_onnx_graph(QP, feed, output_names=go)
    Lp = np.float32(next(np.asarray(x).flatten()[0] for x in r if np.asarray(x).size == 1))
    hb, db = bits(Lp), bits(dev[a])
    ex = hb == db; n_exact += ex
    rel = abs(float(Lp) - float(dev[a])) / (abs(float(dev[a])) + 1e-12)
    print(f"{a:>2} {float(Lp):>12.7f} {hb:>10x} {db:>10x} {str(bool(ex)):>5} {rel:>10.2e}")
print(f"\nSTEP0 BIT-EXACT: {n_exact}/{N_ACCUM}")
