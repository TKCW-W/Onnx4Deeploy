import json, hashlib
import numpy as np

e15 = json.load(open("../exp15_QZO_eco/results.json"))["arms"]
cal = json.load(open("../exp_calibration/results.json"))
recal = cal["incremental"]["qzo"]   # direct@1e-5, seed 42 (original reference)

print("=== exp15 direct_1e5 (seed42) vs exp_cal incremental.qzo (seed42) — both direct@1e-5 ===")
print("round | exp15 after | exp_cal after | delta")
for i in range(1, 5):
    a = e15["direct_1e5"][f"round{i}"]["acc_after"]
    b = recal[f"round{i}"]["acc_after"]
    tag = "same" if abs(a - b) < 0.01 else f"{a-b:+.2f}"
    print(f"  {i}   |   {a:5.2f}%   |    {b:5.2f}%   | {tag}")

print("\n=== direct@1e-5 MULTISEED (incremental_multiseed) — acc_after by seed ===")
ms = cal.get("incremental_multiseed", {})
for seed, arms in ms.items():
    q = arms.get("qzo", {})
    row = [q.get(f"round{i}", {}).get("acc_after") for i in range(1, 5)]
    print(f"  {seed}: b2={row[0]}, b3={row[1]}, b4={row[2]}, b5={row[3]}")

print("\n=== FT-data identity: single cache both exp15 arms + reference load ===")
d = np.load("../exp_calibration/data_cache_incr.npz")
for r_ in [1, 2, 3, 4]:
    h = hashlib.md5(d[f"trX{r_}"].tobytes()).hexdigest()[:12]
    print(f"  round{r_} trX {d[f'trX{r_}'].shape} md5={h}")
