# SPDX-License-Identifier: MIT
"""exp13 micro-trace: build the step-0 +eps parameter set (conv int params as-is; the 12 fp32 params perturbed with
seed 42 / node idx exactly as the frozen train graph does) in extract_qzo_weights key format, and record the host
train-graph step-0 +eps logits/loss (window 0) as the equivalence target. Run in agitated_hugle."""
import struct, sys, numpy as np, onnx
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import _iter_qzo_params
from onnx4deeploy.transform.qzo_weight_integerize import build_int8_forward as _bif
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph, _perturb_rademacher
E = "/app/Onnx4Deeploy/QZO_exp"; FIX = f"{E}/exp13/baked_3e6_freeze"; EPS, SEED = 0.01, 42
m = onnx.load(f"{FIX}/network_zo_train.onnx"); names = [i.name for i in m.graph.input]
inp = np.load(f"{FIX}/inputs.npz"); P = {n: inp[f"arr_{j:04d}"] for j, n in enumerate(names) if f"arr_{j:04d}" in inp and n not in ("input", "label")}
mm, ms = _bif(onnx.load(f"{FIX}/network.onnx")); pmeta = list(_iter_qzo_params(mm, ms))
dump = {}
for p in pmeta:
    nm = p["name"]
    if p["kind"] == "rqs": dump[nm] = np.asarray(P[nm])
    else: dump[nm] = _perturb_rademacher(np.asarray(P[nm]).astype(np.float32), SEED, p["idx"], EPS, 1).reshape(P[nm].shape).astype(np.float32)
    print(f"  {nm:32s} kind={p['kind']:5s} idx={p['idx']:2d} {'perturbed +eps' if p['kind']!='rqs' else 'as-is'}")
np.savez(f"{E}/exp13/micro/step0_peps_params.npz", **dump)
d = np.load(f"{E}/exp_calibration/data_cache_incr.npz"); X, Y = d["trX1"], d["trY1"]
feed = {"input": X[0:1].astype(np.float32), "label": np.asarray(Y[0]).reshape(1, 1).astype(np.int64), **P}
logits, loss = run_onnx_graph(f"{FIX}/network_zo_train.onnx", feed, output_names=["output", "loss"])
bits = lambda f: struct.unpack(">I", struct.pack(">f", np.float32(f)))[0]
np.savez(f"{E}/exp13/micro/step0_train_logits.npz", logits=np.asarray(logits, np.float32), loss=np.float32(np.asarray(loss).flatten()[0]))
print("train-graph step-0 +eps window0: loss bits", f"{bits(np.asarray(loss).flatten()[0]):08x}", "(expect 3eb41bcd)  logits bits", [f"{bits(v):08x}" for v in np.asarray(logits).flatten()])
