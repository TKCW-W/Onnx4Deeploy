# SPDX-License-Identifier: MIT
"""exp11: direct-int8 vs master-weight update under an identical ~1e-7 loss perturbation (delta).
Compares the unperturbed (inj0) and perturbed (inj1) trajectories per mode: per-step relative loss
difference (categorized like the device-vs-host table) and int8 weight disagreement at snapshots."""
import sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "/Users/qiwenwu/ETH/Onnx4Deeploy/QZO_exp/exp11/surrogate"
def load(m, i):
    try: return np.load(f"{D}/{m}_inj{i}.npz")
    except Exception: return None
for mode in ["direct", "master"]:
    a, b = load(mode, 0), load(mode, 1)
    if a is None or b is None: print(f"{mode}: (no data yet)"); continue
    n = min(len(a["lp"]), len(b["lp"])); la, lb = a["lp"][:n].astype(np.float64), b["lp"][:n].astype(np.float64)
    rel = (np.abs(la - lb) / (np.abs(la) + 1e-12)).mean(1)                       # per step, mean over 4 accum
    bit = (a["lp"][:n].view(np.uint32) == b["lp"][:n].view(np.uint32)).all(1)
    print(f"\n=== {mode.upper()} update, inj0 vs inj1 (delta~3e-7): {n} steps ===")
    print(f"{'steps':>10} {'bit-exact':>9} {'ulp(<1e-5)':>10} {'LARGE(>=1e-3)':>13} {'median rel':>11} {'max rel':>9}")
    for lo, hi in [(0, 25), (25, 100), (100, 200), (200, 300), (300, 450), (450, 600)]:
        if lo >= n: break
        hi = min(hi, n); r = rel[lo:hi]; bt = bit[lo:hi]
        print(f"{lo:4d}-{hi:4d} {100*bt.mean():8.1f}% {100*((~bt)&(r<1e-5)).mean():9.1f}% {100*(r>=1e-3).mean():12.1f}% {np.median(r):11.2e} {r.max():9.2e}")
    # int8 weight disagreement at common snapshots
    su = [u for u in a["snap_u"] if u in set(b["snap_u"].tolist())]
    ia = {u: i for i, u in enumerate(a["snap_u"])}; ib = {u: i for i, u in enumerate(b["snap_u"])}
    line = "  int8 weights differing (of %d) at u=" % a["snap"].shape[1]
    line += ", ".join(f"{u}:{int((a['snap'][ia[u]] != b['snap'][ib[u]]).sum())}" for u in su[:14])
    print(line)
    if mode == "master" and len(a["msnap"]) and len(b["msnap"]):
        print("  master |diff| max at u=" + ", ".join(f"{u}:{np.abs(a['msnap'][ia[u]]-b['msnap'][ib[u]]).max():.2e}" for u in su[:14]))
