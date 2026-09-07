# SPDX-License-Identifier: MIT
"""Independent device-vs-host check for a QZO round-1 GVSoC log (host-side, no container needed).
Usage: python3 verify_bitexact.py <device_round1.log[.gz]> <host outputs.npz>
Decodes every logged L+/L- as raw IEEE-754 bits (`lp_bits=0x..`, `lm_bits=0x..`), applies the harness rule
|device - ref| > 0.001 abs (validated to reproduce the device's own `Errors:` count exactly), and prints the per-band
bit-exact / ulp-only / LARGE breakdown and the first LARGE step."""
import gzip, re, struct, sys
import numpy as np
log, ref = sys.argv[1], sys.argv[2]
txt = (gzip.open(log, "rt", errors="ignore") if log.endswith(".gz") else open(log, errors="ignore")).read()
R = np.load(ref); tot_err = tot_n = 0
for key, rk in (("lp_bits", "loss_plus"), ("lm_bits", "loss_minus")):
    d = np.array([struct.unpack(">f", bytes.fromhex(h))[0] for h in re.findall(key + r"=0x([0-9a-fA-F]{8})", txt)], np.float32)
    r = R[rk].astype(np.float32); n = min(len(d), len(r)); d, r = d[:n], r[:n]
    ad = np.abs(d.astype(np.float64) - r); err = int((ad > 1e-3).sum()); tot_err += err; tot_n += n
    rel = ad / (np.abs(r) + 1e-12); bit = d.view(np.uint32) == r.view(np.uint32)
    S = n // 4; rs = rel[:S * 4].reshape(S, 4); bs = bit[:S * 4].reshape(S, 4).all(1); large = (rs >= 1e-3).any(1)
    print(f"{rk:10s}: n={n}  |diff|>0.001: {err}  first LARGE step: {int(np.argmax(large)) if large.any() else None}")
    print(f"  {'steps':>10} {'bit-exact':>9} {'ulp-only':>9} {'LARGE':>6} {'median rel':>11}")
    for lo, hi in [(0, 25), (25, 100), (100, 300), (300, 600), (600, 1200), (1200, S)]:
        if lo >= S: break
        hi = min(hi, S); rr = rs[lo:hi].mean(1); bb = bs[lo:hi]; LL = large[lo:hi]
        print(f"  {lo:4d}-{hi:4d} {100*bb.mean():8.1f}% {100*((~bb)&(rr<1e-5)).mean():8.1f}% {100*LL.mean():5.1f}% {np.median(rr):11.2e}")
m = re.search(r"Errors:\s*(\d+)\s*out of\s*(\d+)", txt)
print(f"\nharness printed line : {m.group(0) if m else '(none)'}   <- only meaningful if the fixture was packed with THIS reference")
print(f"recount (harness rule): {tot_err} out of {tot_n}")
