# SPDX-License-Identifier: MIT
"""exp15 summary figure: accuracy vs conv-weight movement per arm, + the ablation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "results.json"))["arms"]

LABELS = {
    "direct_3e6": "direct INT8\n(stall, 3e-6)",
    "sgdm_none_3e6_b90": "SGDM no-EF\n(ablation)",
    "eco_mf_3e6_b99": "ECO mf β0.99\n(α too weak)",
    "eco_mf_3e6_b90": "ECO mf β0.9\n(RTN)",
    "eco_mf_3e6_b90_sr": "ECO mf β0.9\n(SR)",
    "eco_ex_3e6_b90": "ECO exact-EF\nβ0.9",
    "direct_1e5": "direct INT8\n(ref, 1e-5)",
}
order = ["direct_3e6", "sgdm_none_3e6_b90", "eco_mf_3e6_b99", "eco_mf_3e6_b90",
         "eco_mf_3e6_b90_sr", "eco_ex_3e6_b90", "direct_1e5"]
acc = [np.mean([r[a][f"round{i}"]["acc_after"] for i in range(1, 5)]) for a in order]
union = [r[a]["round4"]["convw_union_ever_pct"] for a in order]
is_eco = ["eco" in a for a in order]
colors = ["#c44" if a == "direct_3e6" else "#999" if a in ("sgdm_none_3e6_b90", "eco_mf_3e6_b99")
          else "#2f6fdb" if "eco" in a else "#1b7f4b" for a in order]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

# left: mean accuracy bars with movement annotation
xs = np.arange(len(order))
ax[0].bar(xs, acc, color=colors)
ax[0].axhline(86.11, color="#1b7f4b", ls="--", lw=1, label="references (direct@1e-5 / float-ZO) 86.11")
ax[0].axhline(85.42, color="#c44", ls=":", lw=1, label="BN-only floor (conv frozen) 85.42")
ax[0].axhline(85.56, color="#888", ls=":", lw=0.8, label="zero-shot 85.56")
for x, a_, u in zip(xs, acc, union):
    ax[0].text(x, a_ + 0.05, f"{a_:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax[0].text(x, 84.2, f"{u:.0f}%\nmoved", ha="center", va="bottom", fontsize=7, color="#333")
ax[0].set_xticks(xs)
ax[0].set_xticklabels([LABELS[a] for a in order], fontsize=8)
ax[0].set_ylim(84.0, 87.4)
ax[0].set_ylabel("mean post-FT balanced accuracy (%)  — 4 rounds")
ax[0].set_title("ECO breaks the INT8 conv stall at lr 3e-6 (no master weights)")
ax[0].legend(fontsize=8, loc="lower right")
ax[0].grid(axis="y", alpha=.3)

# right: accuracy vs write-sparsity trade-off
ax[1].scatter(union, acc, s=90, c=colors, zorder=3)
# stagger label offsets so coincident points (SR/exact at ~100%; ablation/β0.99 at 0%) don't overlap
yoff = {"eco_mf_3e6_b90_sr": 6, "eco_ex_3e6_b90": -14, "sgdm_none_3e6_b90": 6,
        "eco_mf_3e6_b99": -14}
xoff = {"eco_mf_3e6_b90_sr": -70, "eco_ex_3e6_b90": -80, "sgdm_none_3e6_b90": 6,
        "eco_mf_3e6_b99": 6}
for a_, u, ac in zip(order, union, acc):
    ax[1].annotate(LABELS[a_].replace("\n", " "), (u, ac), fontsize=7,
                   xytext=(xoff.get(a_, 4), yoff.get(a_, 4)), textcoords="offset points")
ax[1].axhline(86.11, color="#1b7f4b", ls="--", lw=1)
ax[1].axhline(85.42, color="#c44", ls=":", lw=1)
ax[1].set_xlabel("conv weights ever updated over 4 rounds (%)  →  write traffic")
ax[1].set_ylabel("mean post-FT balanced accuracy (%)")
ax[1].set_title("Accuracy vs write-sparsity: memory-free RTN is write-sparse")
ax[1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(HERE / "eco_summary.png", dpi=140)
print("wrote", HERE / "eco_summary.png")
