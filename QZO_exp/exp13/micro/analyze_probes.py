# SPDX-License-Identifier: MIT
"""exp13 micro-trace: per-probe device-vs-host summary (bit-level, OUTPUT_TOL=0): non-identical elements, ulp histogram."""
import glob, re, struct, numpy as np
f = lambda u: struct.unpack(">f", struct.pack(">I", u))[0]
print(f"{'probe':8s} {'tensor':46s} {'n_diff/n':>16s} {'%':>6s} {'1ulp':>6s} {'2ulp':>6s} {'>2ulp':>6s} {'maxulp':>6s} {'max|d|':>9s}")
for log in sorted(glob.glob("qzo13_probe_*.log")):
    t = open(log, errors="ignore").read(); k = log[12:14]
    node = next((l.split("|")[0].split()[2] for l in open("probe_results.txt") if f"probe_{int(k)} " in l), "?")
    m = re.search(r"Errors: (\d+) out of (\d+)", t)
    if not m: print(f"{k:8s} {node[:46]:46s} (no verdict)"); continue
    n, N = int(m.group(1)), int(m.group(2))
    pairs = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"BITS idx=\d+ exp=0x([0-9a-f]{8}) act=0x([0-9a-f]{8})", t)]
    ul = np.array([abs(a - b) for a, b in pairs]) if pairs else np.zeros(0, int)
    md = max((abs(f(a) - f(b)) for a, b in pairs), default=0.0)
    print(f"{k:8s} {node[:46]:46s} {n:8d}/{N:<7d} {100*n/N:6.2f} {int((ul==1).sum()):6d} {int((ul==2).sum()):6d} {int((ul>2).sum()):6d} {int(ul.max()) if ul.size else 0:6d} {md:9.2e}")
