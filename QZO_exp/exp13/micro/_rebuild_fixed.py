import sys, subprocess, numpy as np, onnx
sys.path.insert(0, "/app/Onnx4Deeploy")
E="/app/Onnx4Deeploy/QZO_exp"; M=f"{E}/exp13/micro"
d=dict(np.load(f"{M}/step0_peps_params.npz"))
for k in list(d):
    if k.endswith("bias_rqsadd"): d[k]=(d[k].astype(np.int64)-32768).astype(np.int32)   # raw bias: strip the baked div/2 (div=65536 all blocks)
np.savez(f"{M}/step0_peps_params_rawbias.npz", **d); print("raw-bias params written; block0 bias now", d["blocks.0.conv.bias_rqsadd"][:3])
r=subprocess.run(f"cd /app/TrainDeeploy/DeeployTest && python3 experiments/deliverable/exp9_QZO_round1/build_qzo_infer_fixture.py --fixture-dir {E}/exp13/baked_3e6_freeze --dump-npz {M}/step0_peps_params_rawbias.npz --out-dir {M}/qinfer_step0_fixed --windows 1", shell=True, capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "injected" in l or "accuracy" in l])
# host layer-wise re-check
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph, freeze_conv_pmul
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
OPS=("Quant","Conv","RequantShift","Dequant","BatchNormInternal","Relu","MaxPool","GlobalAveragePool","Gemm")
pi=np.load(f"{M}/probe_input.npz"); X=pi["input"]; Yl=pi["label"]
QP="/tmp/_lw2_p.onnx"; _, P = build_qzo_train_graph(f"{E}/exp12/baked_3e6/network.onnx", QP, eps=+0.01, seed=42); m=onnx.load(QP); freeze_conv_pmul(m); onnx.save(m,QP)
IG=f"{M}/qinfer_step0_fixed/network.onnx"; mi=onnx.load(IG)
ct=[(n.op_type,n.output[0]) for n in m.graph.node if n.op_type in OPS]; ci=[(n.op_type,n.output[0],n.name) for n in mi.graph.node if n.op_type in OPS]
vt=run_onnx_graph(QP,{"input":X,"label":Yl,**P},output_names=[t for _,t in ct]); vi=run_onnx_graph(IG,{"input":X},output_names=[t for _,t,_ in ci])
bad=[]
for (op,tn),(_,_,nn),a,b in zip(ct,ci,vt,vi):
    a=np.asarray(a); b=np.asarray(b)
    diff = int((a!=b).sum()) if a.dtype.kind in "iu" else int((a.astype(np.float32).view(np.uint32)!=b.astype(np.float32).view(np.uint32)).sum())
    if diff: bad.append((op,nn,diff))
print("HOST train-vs-fixed-infer: tensors differing =", len(bad), "of", len(ct), bad[:3])
print("PROBE NODE NAMES (fp32 points):"); [print("  ",op,nn) for op,_,nn in ci if op in ("Dequant","BatchNormInternal","MaxPool","GlobalAveragePool","Gemm")]
