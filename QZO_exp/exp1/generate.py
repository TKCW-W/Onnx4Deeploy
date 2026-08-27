# QZO exp1 — SpeechNet int8 quantized-ZO fixture generator (weights-as-inputs, BatchNormInternal, Add/Sub).
import numpy as np, torch, collections, os
from brevitas.graph.calibrate import calibration_mode
from onnx4deeploy.models.pytorch_models.speechnet.speechnet_quant import QuantSpeechNetDeploy
from onnx4deeploy.transform.quant_scale_dump import dump_brevitas_scales
from onnx4deeploy.transform.zo_transform import generate_zo_graph
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
# QZO graph THROUGH generate_zo_graph (build_qzo_int8_graph + shared append_cross_entropy_loss)
param_inputs=generate_zo_graph(inference_onnx=None, output_onnx=f"{OUT}/network_zo_train.onnx",
                               zo_config={"epsilon":0.01,"seed":42}, noise_type="rqs_rademacher",
                               qzo_model=m, qzo_scales=scales)
g=onnx.load(f"{OUT}/network_zo_train.onnx").graph
print("OPS:", dict(collections.Counter(n.op_type for n in g.node)))
print("GRAPH INPUTS:", [i.name for i in g.input])
# reference (perturbation ON) — weights fed as INPUTS
inp=np.random.randn(1,1,14,700).astype(np.float32); lab=np.random.randint(0,9,(1,1)).astype(np.int64)
feed={"input":inp,"label":lab,**param_inputs}
out=np.asarray(run_onnx_graph(f"{OUT}/network_zo_train.onnx", feed, output_names=["log_prob"])[0])
np.savez(f"{OUT}/inputs.npz", input=inp, label=lab, **param_inputs); np.savez(f"{OUT}/outputs.npz", output=out)
print("REF(perturb on): argmax", int(np.argmax(out)))
# numerical sanity: eps=0 (no perturbation) vs Brevitas forward, SAME input
build_qzo_int8_graph(m, scales, f"{OUT}/_eps0.onnx", eps=0.0, seed=42)
o0=np.asarray(run_onnx_graph(f"{OUT}/_eps0.onnx", feed, output_names=["output"])[0])
with torch.no_grad(): bl=m(torch.from_numpy(inp)).numpy()
print("SANITY: int8(eps=0) argmax", int(np.argmax(o0)), "| Brevitas argmax", int(np.argmax(bl)),
      "| MATCH", int(np.argmax(o0))==int(np.argmax(bl)))
os.remove(f"{OUT}/_eps0.onnx")
