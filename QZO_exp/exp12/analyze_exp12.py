# SPDX-License-Identifier: MIT
"""exp12 analysis: device (lr 3e-6) vs host reference, side by side with the 1e-5 run.
Reports (1) the harness's own `Errors: N out of M` line, (2) the same count recomputed from the raw
lp_bits/lm_bits vs the reference with the harness rule |dev - ref| > TOL (0.001 abs), (3) per-band
categories (bit-exact / ulp / LARGE) for L+ and L-, (4) first LARGE step."""
import re, struct, sys, numpy as np
E = "/Users/qiwenwu/ETH/Onnx4Deeploy/QZO_exp"
def bits(log, key):
    return np.array([struct.unpack(">f", bytes.fromhex(h))[0] for h in re.findall(key + r"=0x([0-9a-fA-F]{8})", open(log, errors="ignore").read())], np.float32)
def run(name, log, ref_npz, TOL=1e-3):
    txt = open(log, errors="ignore").read(); m = re.search(r"Errors:\s*(\d+)\s*out of\s*(\d+)", txt)
    print(f"\n########## {name} ##########")
    print("harness line     :", m.group(0) if m else "(no Errors: line yet)")
    ref = np.load(ref_npz); tot_err = 0; tot_n = 0
    for key, rk in (("lp_bits", "loss_plus"), ("lm_bits", "loss_minus")):
        d = bits(log, key); r = ref[rk].astype(np.float32); n = min(len(d), len(r)); d, r = d[:n], r[:n]
        if n == 0: print(f"  {key}: no data"); continue
        ad = np.abs(d.astype(np.float64) - r); err = int((ad > TOL).sum()); tot_err += err; tot_n += n
        rel = ad / (np.abs(r) + 1e-12); bit = d.view(np.uint32) == r.view(np.uint32)
        S = n // 4; rs = rel[:S*4].reshape(S, 4); bs = bit[:S*4].reshape(S, 4).all(1); large = (rs >= 1e-3).any(1)
        first = int(np.argmax(large)) if large.any() else None
        print(f"  {rk:10s}: n={n:5d}  |diff|>{TOL}: {err:5d} ({100*err/n:5.1f}%)   first LARGE step: {first}")
        print(f"    {'steps':>10} {'bit-exact':>9} {'ulp(<1e-5)':>10} {'LARGE':>7} {'median rel':>11}")
        for lo, hi in [(0,25),(25,100),(100,300),(300,600),(600,1200),(1200,2700)]:
            if lo >= S: break
            hi = min(hi, S); rr = rs[lo:hi].mean(1); bb = bs[lo:hi]; LL = large[lo:hi]
            print(f"    {lo:4d}-{hi:4d} {100*bb.mean():8.1f}% {100*((~bb)&(rr<1e-5)).mean():9.1f}% {100*LL.mean():6.1f}% {np.median(rr):11.2e}")
    if tot_n: print(f"  recomputed harness count: {tot_err} out of {tot_n}")
run("lr 1e-5 (exp10 round-1)", f"{E}/exp10/baked200_device_round1.log", f"{E}/exp10/baked_200ep/outputs.npz")
log3 = sys.argv[1] if len(sys.argv) > 1 else f"{E}/exp12/device_round1_3e6.log"
try: run("lr 3e-6 (exp12 round-1)", log3, f"{E}/exp12/baked_3e6/outputs.npz")
except FileNotFoundError as e: print("\n(3e-6 run not available yet:", e, ")")
