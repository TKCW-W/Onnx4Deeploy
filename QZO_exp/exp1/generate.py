# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""QZO exp1 — generate the SpeechNet quantized-ZO fixture + scales JSON.

Flow: build+calibrate QuantSpeechNet → dump per-channel scales → export QCDQ base (Export4Deeploy) →
qzo_transform (online weight-quant + int8 RQSPerturb + SCE loss) → run_onnx_graph reference.
Run in agitated_hugle:  PYTHONPATH=/app/Onnx4Deeploy:/app/Onnx4Deeploy/DeepQuant python3 generate.py
"""
import os, collections
import numpy as np, torch, onnx
from brevitas.graph.calibrate import calibration_mode

from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
from onnx4deeploy.transform.quant_scale_dump import dump_brevitas_scales
from onnx4deeploy.transform.qzo_transform import generate_qzo_train_graph
from DeepQuant.Export4Deeploy import exportBrevitas

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); np.random.seed(0)

# 1. Brevitas model + PTQ calibration
m = QuantSpeechNetDeploy(num_classes=9); m.eval()
for n, p in m.named_parameters():
    if "weight" in n and p.dim() > 1: torch.nn.init.normal_(p, 0.0, 0.05)
    if "bias" in n: torch.nn.init.uniform_(p, 0.01, 0.02)
with torch.no_grad(), calibration_mode(m):
    m(torch.randn(8, 1, 14, 700))

# 2. scales JSON (single source of truth)
dump_brevitas_scales(m, os.path.join(OUT, "speechnet_scales.json"))

# 3. QCDQ base graph
om = exportBrevitas(m, torch.randn(1, 1, 14, 700), debug=False)
onnx.save(om, os.path.join(OUT, "network_infer.onnx"))
print("  [gen] base ops:", dict(collections.Counter(n.op_type for n in om.graph.node)))

# 4. QZO train graph (online weight-quant + int8 perturb + loss)
generate_qzo_train_graph(os.path.join(OUT, "network_infer.onnx"),
                         os.path.join(OUT, "network_zo_train.onnx"),
                         {"epsilon": 0.01, "seed": 42},
                         scales_path=os.path.join(OUT, "speechnet_scales.json"))
zt = onnx.load(os.path.join(OUT, "network_zo_train.onnx"))
print("  [gen] zo_train ops:", dict(collections.Counter(n.op_type for n in zt.graph.node)))

# 5. reference via pure-Python executor (toposort first — custom ops may be unordered)
def toposort(mp):
    g = mp.graph; avail = {i.name for i in g.input} | {i.name for i in g.initializer}
    nodes = list(g.node); out = []
    while nodes:
        prog = False
        for n in list(nodes):
            if all((x in avail) or x == "" for x in n.input):
                out.append(n); nodes.remove(n)
                avail.update(n.output); prog = True
        if not prog: break
    del g.node[:]; g.node.extend(out); return mp
onnx.save(toposort(onnx.load(os.path.join(OUT, "network_zo_train.onnx"))), os.path.join(OUT, "_zt_sorted.onnx"))

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
inp = np.random.randn(1, 1, 14, 700).astype(np.float32)
label = np.random.randint(0, 9, (1, 1)).astype(np.int64)
try:
    out = np.asarray(run_onnx_graph(os.path.join(OUT, "_zt_sorted.onnx"), {"input": inp, "label": label}))
    np.savez(os.path.join(OUT, "inputs.npz"), input=inp, label=label)
    np.savez(os.path.join(OUT, "outputs.npz"), output=out)
    print(f"  [gen] reference OK: outputs.npz shape={out.shape}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  [gen] reference FAILED: {type(e).__name__}: {str(e)[:120]}")
os.remove(os.path.join(OUT, "_zt_sorted.onnx"))
print("  [gen] DONE")
