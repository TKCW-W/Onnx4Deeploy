import struct, sys, numpy as np, onnx
from onnx import numpy_helper as nh
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph, freeze_conv_pmul
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
E="/app/Onnx4Deeploy/QZO_exp"; OPS=("Quant","Conv","RequantShift","Dequant","BatchNormInternal","Relu","MaxPool","GlobalAveragePool","Gemm")
pi=np.load(f"{E}/exp13/micro/probe_input.npz"); X=pi["input"]; Yl=pi["label"]
QP="/tmp/_lw_p.onnx"; _, P = build_qzo_train_graph(f"{E}/exp12/baked_3e6/network.onnx", QP, eps=+0.01, seed=42)
m=onnx.load(QP); freeze_conv_pmul(m); onnx.save(m,QP)
IG=f"{E}/exp13/micro/qinfer_step0/network.onnx"; mi=onnx.load(IG)
chain=lambda g: [(n.op_type, n.output[0]) for n in g.graph.node if n.op_type in OPS]
ct, ci = chain(m), chain(mi)
print("op sequence identical:", [o for o,_ in ct]==[o for o,_ in ci], "| train ops", len(ct), "infer ops", len(ci))
vt = run_onnx_graph(QP, {"input":X,"label":Yl,**P}, output_names=[t for _,t in ct])
vi = run_onnx_graph(IG, {"input":X}, output_names=[t for _,t in ci])
first=None
for (op,tn),(_,tn2),a,b in zip(ct,ci,vt,vi):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: print(f"{op:18s} SHAPE {a.shape} vs {b.shape}"); first=first or op; continue
    if a.dtype.kind in "iu": d=int((a!=b).sum()); s=f"int diff elems={d}/{a.size}"
    else:
        d=int((a.astype(np.float32).view(np.uint32)!=b.astype(np.float32).view(np.uint32)).sum()); s=f"fp32 bit-diff elems={d}/{a.size} max|d|={np.abs(a.astype(np.float64)-b).max():.3e}"
    print(f"{op:18s} {tn[:40]:40s} {s}"); 
    if d and first is None: first=(op,tn)
print("FIRST DIFFERENCE:", first)
init=lambda g,suf:[ (i.name, nh.to_array(i).reshape(-1)[:3]) for i in g.graph.initializer if i.name.endswith(suf)]
print("train RequantShift add-inputs:", [(n.input[2], nh.to_array({i.name:i for i in m.graph.initializer}[n.input[2]]).reshape(-1)[:2]) for n in m.graph.node if n.op_type=="RequantShift"][:2] if all(n.input[2] in {i.name for i in m.graph.initializer} for n in m.graph.node if n.op_type=="RequantShift") else "add is a graph INPUT (perturbable bias)")
print("infer RequantShift inputs:", [(list(n.input)) for n in mi.graph.node if n.op_type=="RequantShift"][:1])
ii={i.name:i for i in mi.graph.initializer}
for n in [n for n in mi.graph.node if n.op_type=="RequantShift"][:1]:
    for x in n.input[1:]:
        if x in ii: print("  infer", x, nh.to_array(ii[x]).reshape(-1)[:4], nh.to_array(ii[x]).dtype)
print("train bias input P[blocks.0.conv.bias_rqsadd][:4] =", np.asarray(P["blocks.0.conv.bias_rqsadd"]).reshape(-1)[:4])
