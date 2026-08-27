# QZO exp2 — real-data int8 quantized-ZO fixture generator (offline weights-as-inputs).
# Built from the REAL-data -mode quant export (qinfer/network.onnx: fold_3 pretrained,
# S01/session3/vocalized). Produces zo_train (L+/L-) + zo_update + inputs/outputs.
import os, numpy as np, onnx
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph, build_qzo_update_graph
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

OUT = os.path.dirname(os.path.abspath(__file__))
QINFER = f"{OUT}/qinfer/network.onnx"
EPS, SEED = 0.01, 42

# real int8 input (from the real-data calibration export) + label = reference argmax (self-consistent)
inp = np.load(f"{OUT}/qinfer/inputs.npz")["input"]
ref_logits = np.load(f"{OUT}/qinfer/outputs.npz")["output"].flatten()
label = np.array([[int(np.argmax(ref_logits))]], dtype=np.int64)

# zo_train (+eps = L+, weights-as-inputs)
train_path, pin = build_qzo_train_graph(QINFER, f"{OUT}/network_zo_train.onnx", eps=EPS, seed=SEED)
# zo_update (same idx/seed/mul)
upd_path, pin_u = build_qzo_update_graph(QINFER, f"{OUT}/network_zo_update.onnx", eps=EPS, seed=SEED)
assert set(pin) == set(pin_u), "train/update param sets differ"

go = [o.name for o in onnx.load(train_path).graph.output]

def run_loss(eps_signed):
    p, _ = build_qzo_train_graph(QINFER, "/tmp/qzo_pm.onnx", eps=eps_signed, seed=SEED)
    res = run_onnx_graph("/tmp/qzo_pm.onnx", {"input": inp, "label": label, **pin}, output_names=go)
    loss = float(next(np.asarray(r).flatten()[0] for r in res if np.asarray(r).size == 1))
    lp = next(np.asarray(r) for r in res if np.asarray(r).size > 1)
    return loss, lp

Lp, lp_p = run_loss(+EPS)   # L+  (theta + eps z)
Lm, lp_m = run_loss(-EPS)   # L-  (theta - eps z)
grad = (Lp - Lm) / (2 * EPS)

# updated params (perturb direction) from zo_update, +eps
res_u = run_onnx_graph(upd_path, {**pin_u}, output_names=[f"{k}_updated" for k in pin_u])
updated = {f"updated_{k}": np.asarray(v) for k, v in zip(pin_u, res_u)}

np.savez(f"{OUT}/inputs.npz", input=inp, label=label, **pin)
np.savez(f"{OUT}/outputs.npz", loss_plus=np.float32(Lp), loss_minus=np.float32(Lm),
         grad=np.float32(grad), log_prob=lp_p, **updated)
print(f"\n=== QZO exp2 real-data fixture ===")
print(f"  zo_train: {train_path}")
print(f"  zo_update: {upd_path}")
print(f"  #params(inputs): {len(pin)}  graph inputs: {len(onnx.load(train_path).graph.input)}")
print(f"  L+ = {Lp:.6f}   L- = {Lm:.6f}   grad=(L+-L-)/2eps = {grad:.6f}")
print(f"  log_prob argmax {int(np.argmax(lp_p))} (label {int(label[0,0])})")
print(f"  saved inputs.npz, outputs.npz")
