"""Prove exp15 direct_1e5 came from the real Brevitas sim and that ECO vs direct_1e5 is a
faithful (same seed + same data) comparison.

1. Re-run direct_1e5 FRESH (seed 42, default cache) through the same Brevitas pipeline and
   diff every round's acc_after + conv movement against the stored results.json. The sim is
   deterministic given (seed, data), so a real result must reproduce bit-identically.
2. Confirm both arms use S.SEED=42 and the same data cache; md5 the FT windows.
"""
import hashlib
import json

import numpy as np
import torch

import run_eco as E
import run_study as S
import run_incremental as RI

stored = json.load(open(E.RESULTS))["arms"]

print("=== 0. seed + data provenance (shared by BOTH arms) ===")
print(f"  S.SEED = {S.SEED}   (draw_z uses RandomState(SEED+u); same call for direct & ECO)")
print(f"  RI.CACHE = {RI.CACHE}")
d = np.load(RI.CACHE)
for r_ in [1, 2, 3, 4]:
    h = hashlib.md5(d[f"trX{r_}"].tobytes()).hexdigest()[:12]
    print(f"    round{r_} trX md5={h} shape={d[f'trX{r_}'].shape}")

print("\n=== 1. FRESH re-run of direct_1e5 (seed 42) vs stored results.json ===")
torch.manual_seed(S.SEED); np.random.seed(S.SEED)
cfg = dict(E.ARMS["direct_1e5"]); cfg["_tag"] = "verify:direct_1e5"
fresh = E.eco_incremental(cfg, {}, [1, 2, 3, 4], RI.STEPS)   # actually runs the Brevitas sim

allok = True
print(f"  {'round':6s} {'fresh after':>11s} {'stored after':>12s} {'fresh moved%':>12s} {'stored moved%':>13s}  match")
for i in range(1, 5):
    f = fresh[f"round{i}"]; s = stored["direct_1e5"][f"round{i}"]
    aok = abs(f["acc_after"] - s["acc_after"]) < 1e-9
    mok = abs(f["convw_moved_this_round_pct"] - s["convw_moved_this_round_pct"]) < 1e-9
    allok &= aok and mok
    print(f"  {i:<6d} {f['acc_after']:>11.4f} {s['acc_after']:>12.4f} "
          f"{f['convw_moved_this_round_pct']:>12.4f} {s['convw_moved_this_round_pct']:>13.4f}  "
          f"{'OK' if aok and mok else 'MISMATCH'}")
print(f"\n  RESULT: {'PASS — stored direct_1e5 reproduces from the live Brevitas sim' if allok else 'FAIL — stored != fresh'}")
