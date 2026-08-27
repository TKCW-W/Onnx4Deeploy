# QZO exp1 — SpeechNet int8 quantized-ZO fixture generator.
import numpy as np, torch, collections, os
from brevitas.graph.calibrate import calibration_mode
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
from onnx4deeploy.transform.quant_scale_dump import dump_brevitas_scales
from onnx4deeploy.transform.qzo_transform import build_qzo_int8_graph
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
import onnx
OUT=os.path.dirname(os.path.abspath(__file__)); torch.manual_seed(0); np.random.seed(0)
m=QuantSpeechNetDeploy(num_classes=9); m.eval()
for n,p in m.named_parameters():
    if "weight" in n and p.dim()>1: torch.nn.init.normal_(p,0,0.05)
    if "bias" in n: torch.nn.init.uniform_(p,0.01,0.02)
with torch.no_grad(), calibration_mode(m): m(torch.randn(8,1,14,700))
scales=dump_brevitas_scales(m, f"{OUT}/speechnet_scales.json")
build_qzo_int8_graph(m, scales, f"{OUT}/network_zo_train.onnx", eps=0.01, seed=42)
g=onnx.load(f"{OUT}/network_zo_train.onnx").graph
print("OPS:", dict(collections.Counter(n.op_type for n in g.node)))
# reference (perturbation ON)
inp=np.random.randn(1,1,14,700).astype(np.float32); lab=np.random.randint(0,9,(1,1)).astype(np.int64)
out=np.asarray(run_onnx_graph(f"{OUT}/network_zo_train.onnx", {"input":inp,"label":lab}))
np.savez(f"{OUT}/inputs.npz", input=inp, label=lab); np.savez(f"{OUT}/outputs.npz", output=out)
print("REF(perturb on): argmax", int(np.argmax(out)))
# numerical sanity: eps=0 (no perturbation) vs Brevitas forward, SAME input
build_qzo_int8_graph(m, scales, f"{OUT}/_eps0.onnx", eps=0.0, seed=42)
o0=np.asarray(run_onnx_graph(f"{OUT}/_eps0.onnx", {"input":inp,"label":lab}))
with torch.no_grad(): bl=m(torch.from_numpy(inp)).numpy()
print("SANITY: int8(eps=0) argmax", int(np.argmax(o0)), "| Brevitas argmax", int(np.argmax(bl)),
      "| MATCH", int(np.argmax(o0))==int(np.argmax(bl)))
os.remove(f"{OUT}/_eps0.onnx")
