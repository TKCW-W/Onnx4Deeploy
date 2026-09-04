# SPDX-License-Identifier: MIT
"""exp10 iteration 2 — is the requant TRUNCATION (variable-add bias) the forward-divergence cause?

B's RequantShift truncates (>>d, no rounding) because our conv bias is a VARIABLE add
(perturbable), and the merge pass can only bake the rounding constant for a compile-time-
initializer add. Brevitas A rounds. Hypothesis: this half-LSB-per-output truncation bias,
invisible unperturbed, is amplified under the +-eps perturbation and corrupts L+-L-.

Test: compute B's forward loss on the same 4 mini-batches at the +eps and -eps perturbed
states, TWICE — truncating (default) and rounding (QZO_FORCE_REQUANT_ROUND=1) — and compare
both to A's forward at B's exact perturbed state (A rounds). If rounding makes B match A, the
truncation is confirmed as the corruption; the g's should also align.

Run in agitated_hugle: the two env regimes are separate subprocesses.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EPS, N_ACCUM = 0.01, 4

WORKER = r'''
import sys, numpy as np
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.transform.qzo_transform import build_qzo_train_graph
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
FIX="/app/Onnx4Deeploy/QZO_exp/exp9_full"
import onnx
inp=np.load(FIX+"/inputs.npz")
d=np.load("/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz")
X,Y=d["trX1"],d["trY1"]
def losses(sign):
    out=f"/tmp/_rq_{'p' if sign>0 else 'm'}.onnx"
    build_qzo_train_graph(FIX+"/network.onnx", out, eps=sign*EPS, seed=42)
    m=onnx.load(out); names=[i.name for i in m.graph.input]
    # loss node output is the graph's scalar output
    ls=[]
    for i in range(4):
        feed={n:(X[i:i+1] if n=="input" else (Y[i:i+1].reshape(1,1) if n=="label" else inp[f"arr_{j:04d}"])) for j,n in enumerate(names)}
        r=run_onnx_graph(out, feed, output_names=["loss"])
        ls.append(float(np.asarray(r[0]).flatten()[0]))
    return ls
EPS=0.01
lp=losses(+1); lm=losses(-1)
g=sum(p-m for p,m in zip(lp,lm))/(2*EPS*4)
print("RESULT", __import__("json").dumps(dict(lp=lp, lm=lm, g=g)))
'''


def run(force_round):
    env_prefix = "QZO_FORCE_REQUANT_ROUND=1 " if force_round else ""
    cmd = f"docker exec agitated_hugle bash -lc '{env_prefix}python3 -c \"{WORKER}\"'"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):].strip())
    raise RuntimeError("worker produced no RESULT:\n" + out[-2000:])


def main():
    trunc = run(False)
    rnd = run(True)
    a = json.load(open("/app/Onnx4Deeploy/QZO_exp/exp5_A_vs_B/matched_z_losses.json"))
    print("B forward L+ per mb (truncate, current):", [round(x, 4) for x in trunc["lp"]])
    print("B forward L+ per mb (ROUND):            ", [round(x, 4) for x in rnd["lp"]])
    print("A forward L+ at A-state (exp5):          ", [round(p, 4) for p, _ in a["a_pairs"]])
    print()
    print(f"g_proj  truncate={trunc['g']:.4f}   round={rnd['g']:.4f}   A(exp5)={a['a_g']:.4f}")
    json.dump(dict(truncate=trunc, round=rnd, a_g=a["a_g"]),
              open(HERE / "requant_rounding_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
