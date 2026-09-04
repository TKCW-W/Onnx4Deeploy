# SPDX-License-Identifier: MIT
"""Worker: B's forward L+/L- and g at the +-eps perturbed states. Honors
QZO_FORCE_REQUANT_ROUND via the run_onnx_graph RequantShift. Run in agitated_hugle."""
import json
import sys

import numpy as np
import onnx

sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

FIX = "/app/Onnx4Deeploy/QZO_exp/exp9_full"
EPS = 0.01
inp = np.load(FIX + "/inputs.npz")
d = np.load("/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz")
X, Y = d["trX1"], d["trY1"]


def losses(sign):
    out = f"/tmp/_rq_{'p' if sign > 0 else 'm'}.onnx"
    build_qzo_train_graph(FIX + "/network.onnx", out, eps=sign * EPS, seed=42)
    m = onnx.load(out)
    names = [i.name for i in m.graph.input]
    ls = []
    for i in range(4):
        feed = {}
        for j, n in enumerate(names):
            if n == "input":
                feed[n] = X[i:i + 1].astype(np.float32)
            elif n == "label":
                feed[n] = Y[i:i + 1].reshape(1, 1).astype(np.int64)
            else:
                feed[n] = inp[f"arr_{j:04d}"]
        # compute CE ourselves from log_prob (= log-softmax of logits, shape [1,9]),
        # matching the SoftmaxCrossEntropyLoss node: CE = -log_prob[label]
        lp = np.asarray(run_onnx_graph(out, feed, output_names=["log_prob"])[0]).reshape(-1)
        ls.append(float(-lp[int(Y[i])]))
    return ls


lp = losses(+1)
lm = losses(-1)
g = sum(p - m for p, m in zip(lp, lm)) / (2 * EPS * 4)
print("RESULT " + json.dumps(dict(lp=lp, lm=lm, g=g)))
