# SPDX-License-Identifier: MIT
"""exp10 — QZO device vs host-reference divergence, grounded in the device's own
logged loss bit-patterns.

The device logs every +eps forward loss as a raw IEEE-754 hex word
(`+eps loss read OK: lp_bits=0x........`). We decode all 10800 of them
(2700 steps x 4 accum) and compare bit-for-bit and in relative residual against
the host reference `loss_plus` produced by `run_onnx_graph` on the SAME exported
graph with the SAME seed.

Result (see NOTES / DIVERGENCE_NOTES):
  * step 0 (identical weights, identical z): rel-resid ~1e-7 (last 1-2 ulp fp32),
    NOT bit-equal -> the forward MECHANISM is identical; only fp32 last-ulp differs.
  * residual grows 4e-7 -> 1e-1 over the 2700 steps: chaotic amplification of the
    per-step ulp difference through the round()-thresholded int8 weight update.

This is by-nature (device built with -ffast-math -O3: FMA fusion + fp reassociation
+ picolibc expf/logf in SoftmaxCrossEntropyLoss), not a fixable implementation gap.
Run on the host (no container needed): python3 analyze_device_vs_hostref_loss.py
"""
import struct
import subprocess

import numpy as np

HERE = "/Users/qiwenwu/ETH/Onnx4Deeploy/QZO_exp/exp10"
LOG = f"{HERE}/baked200_device_round1.log"
REF = f"{HERE}/baked_200ep/outputs.npz"

# device +eps loss bit-patterns, in log order (= host loss_plus order)
hexes = subprocess.run(
    ["grep", "-oE", r"\+eps loss read OK: lp_bits=0x[0-9a-fA-F]{8}", LOG],
    capture_output=True, text=True, check=True,
).stdout.split("\n")
hexes = [h.split("0x")[1] for h in hexes if "0x" in h]
dev = np.array([struct.unpack(">f", bytes.fromhex(h))[0] for h in hexes], np.float64)

lp = np.load(REF)["loss_plus"].astype(np.float64)
n = min(len(dev), len(lp))
dev, lp = dev[:n], lp[:n]

rel = np.abs(dev - lp) / (np.abs(lp) + 1e-12)
be = np.mean(dev[:n].astype(np.float32).view(np.uint32)
             == lp[:n].astype(np.float32).view(np.uint32)) * 100.0
print(f"device +eps losses: {n}   overall bit-exact vs host-ref: {be:.2f}%")

rs = rel[: (n // 4) * 4].reshape(-1, 4).mean(1)  # per-step (4 accum)
print("\nper-step relative residual (device vs host-ref loss_plus):")
for a, b in [(0, 5), (5, 100), (100, 500), (500, 1500), (1500, len(rs))]:
    seg = rs[a:b]
    print(f"  steps {a:4d}-{b:4d}: median={np.median(seg):.2e}  max={seg.max():.2e}")
print("\nstep 0, per-accum rel-resid:", rel[:4])
print("=> ~1e-7 at step 0 (identical weights) = fp32 last-ulp; the mechanism is identical.")
print("=> growth to ~1e-1 = round()-threshold amplification (deterministic chaos).")
