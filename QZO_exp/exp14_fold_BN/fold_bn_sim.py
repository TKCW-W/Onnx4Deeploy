# SPDX-License-Identifier: MIT
"""exp14: QZO with BatchNorm FOLDED into the quantized conv (supervisor's design) — PyTorch/Brevitas simulation.
Reuses the exp_masterweight_ft harness verbatim (mw.py copy: build = create model -> real-data calibration -> freeze
scales -> trainable params; run_regime; evaluate) and inserts ONE step: right after the pretrained Brevitas model is
created and BEFORE calibration, fold each block's fp32 BN (running stats, gamma, beta) into the block's QuantConv2d
weight/bias and replace the BN by Identity. Weight scales are then re-derived from the folded weights and activation
scales re-calibrated on the folded model. Trainable params = int8 conv W'/b' (+ fc), no fp32 BN params.
Run in agitated_hugle:  EXTRA=x python3 fold_bn_sim.py   (EXTRA non-empty keeps mw's log in append mode)"""
import json, os, sys, time
os.environ.setdefault("EXTRA", "x")
sys.path.insert(0, "/app/Onnx4Deeploy"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import mw                                                   # the harness (copied), HERE = this dir
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter

FOLD_LOG = []
def fold_bn_(model):
    n = 0
    for b in model.blocks:
        bn, conv = b.bn, b.conv
        if not isinstance(bn, nn.BatchNorm2d): continue
        with torch.no_grad():
            inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)          # gamma / sigma
            W = conv.weight; conv.weight.copy_(W * inv.reshape(-1, 1, 1, 1))
            bias = conv.bias if conv.bias is not None else torch.zeros(W.shape[0])
            new_b = (bias - bn.running_mean) * inv + bn.bias
            if conv.bias is None: conv.bias = nn.Parameter(new_b)
            else: conv.bias.copy_(new_b)
        FOLD_LOG.append(f"   folded block: gamma/sigma in [{float(inv.min()):.3f},{float(inv.max()):.3f}]")
        b.bn = nn.Identity(); n += 1
    return n

_orig = SpeechNetExporter.create_brevitas_model
def _create_folded(self, *a, **k):
    m = _orig(self, *a, **k); n = fold_bn_(m); mw.log(f"[exp14] folded {n} BN layers into conv weight/bias; BN -> Identity"); [mw.log(l) for l in FOLD_LOG]
    return m
SpeechNetExporter.create_brevitas_model = _create_folded

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    model, params, trX, trY, evX, evY, acts0 = mw.build()        # fold happens inside (patched constructor)
    zs = mw.evaluate(model, evX, evY); mw.log(f"[exp14] FOLDED zero-shot: {zs}")
    regimes = [s for s in os.environ.get("REGIMES", "direct@3e-6,direct@1e-5,direct@3e-5,direct@1e-4,master@3e-6,master@1e-5,master@3e-5").split(",") if s]
    out = {"design": "BN folded into QuantConv2d (weight+bias), BN->Identity, recalibrated", "zero_shot": zs,
           "n_trainable": len(params), "kinds": {k: sum(p["kind"] == k for p in params) for k in ("quant", "float")}, "runs": {}}
    for spec in regimes:
        mode, lr = spec.split("@"); lr = float(lr)
        t0 = time.time(); r = mw.run_regime(model, params, trX, trY, evX, evY, mode, lr, tag=spec)
        r["seconds"] = time.time() - t0; out["runs"][spec] = r
        mw.log(f"[exp14] {spec}: bal_acc={r.get('balanced_accuracy', float('nan')):.2f}%  ({r['seconds']:.0f}s)")
        with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(out, f, indent=2, default=float)
    mw.log("[exp14] DONE")

if __name__ == "__main__": main()
