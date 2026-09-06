import struct, sys, numpy as np, onnx
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph, freeze_conv_pmul
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
E="/app/Onnx4Deeploy/QZO_exp"; FIX=f"{E}/exp13/baked_3e6_freeze"
b=lambda f: struct.unpack(">I",struct.pack(">f",np.float32(f)))[0]
pi=np.load(f"{E}/exp13/micro/probe_input.npz"); X=pi["input"]; Yl=pi["label"]
# way 1: fresh build + freeze, name-keyed params (== check_step0_freeze)
QP="/tmp/_eq_p.onnx"; _, P1 = build_qzo_train_graph(f"{E}/exp12/baked_3e6/network.onnx", QP, eps=+0.01, seed=42)
m=onnx.load(QP); freeze_conv_pmul(m); onnx.save(m,QP)
lg1, l1 = run_onnx_graph(QP, {"input":X,"label":Yl,**P1}, output_names=["output","loss"])
# way 2: saved fixture graph + arr_ mapping
m2=onnx.load(f"{FIX}/network_zo_train.onnx"); names=[i.name for i in m2.graph.input]; inp=np.load(f"{FIX}/inputs.npz")
P2={n: inp[f"arr_{j:04d}"] for j,n in enumerate(names) if n not in ("input","label")}
lg2, l2 = run_onnx_graph(f"{FIX}/network_zo_train.onnx", {"input":X,"label":Yl,**P2}, output_names=["output","loss"])
print("way1 fresh-build loss", f"{b(np.asarray(l1).flatten()[0]):08x}", "| way2 saved-graph loss", f"{b(np.asarray(l2).flatten()[0]):08x}", "| P1==P2 by name:", all(np.array_equal(np.asarray(P1[k]),np.asarray(P2[k])) for k in P1 if k in P2), "| keys only in one:", sorted(set(P1)^set(P2))[:4])
# inference graph on the same window
lg3 = run_onnx_graph(f"{E}/exp13/micro/qinfer_step0/network.onnx", {"input":X}, output_names=["output"])[0]
f=lambda a: [f"{b(v):08x}" for v in np.asarray(a,np.float32).flatten()[:9]]
print("train(way1) logits", f(lg1)); print("infer  logits     ", f(lg3)); print("EQUIVALENT:", f(lg1)==f(lg3))
