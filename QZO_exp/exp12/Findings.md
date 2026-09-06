# exp12 — Findings: device vs host-ref at lr 3e-6 (control for exp11)

2026-09-06 21:10 CEST · Branch `feat/QZO` · Commands in `Plan.md` (reproduction section)

## Setup (identical to the 1e-5 round-1 except lr)
Same fixture (verified: both graphs byte-identical and all 133 `inputs.npz` arrays identical to the 1e-5 export —
lr is not baked into the fixture), seed 42, 54 windows, pooled@99.99, n_accum 4, 2700 steps, frozen-BN, `-D
BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`. Device run overlapped with the export; errors recomputed from the logged raw
`lp_bits`/`lm_bits` with the harness rule |dev − ref| > 0.001 abs (validated to reproduce the harness count exactly on
the 1e-5 log: 8302 + 8263 = 16565). The harness's own printed line (`20718`) is against the placeholder reference and is
void. Logs: `device_round1_3e6.log.gz`, `export_3e6.log`; params: `device_weights_3e6.npz`, `baked_3e6/`.

## Result
| | lr 1e-5 | lr 3e-6 |
|---|---|---|
| **errors / 21,600** | **16,565 (76.7%)** | **4,254 (19.7%)** |
| L+ errors / L− errors | 8,302 / 8,263 | 2,125 / 2,129 |
| first LARGE step L+ / L− | 5 (transient) / 326 | 301 / 424 |
| LARGE 0–300 / 300–600 / 600–1200 / 1200–2700 | 0–4% / 67% / 100% / 100% | 0% / 1–3% / ~20% / 68% |
| median rel diff 1200–2700 | 9.8e-2 | 4e-3 |

**Final parameters device vs host at 3e-6** (`extract_qzo_weights.py --ref-outputs`):
| class | result |
|---|---|
| int8 conv weights (5 blocks, 14,880) | **bit-exact** (0 moved on either side) |
| int32 `bias_rqsadd` (104) | block 0 identical; blocks 1–4 **differ**: 13/16, 12/16, 27/32, 29/32, max 4–12 units |
| fp32 BN γ/β + fc (10 + 2 tensors) | all differ, max 1.3e-4 … 2.5e-4 (1e-5 run: ~3e-3) |
Host-side movement at 3e-6: conv weights 0/14,880; bias entries 95/104 (2,486 units) — **the bias int path is NOT
inert at 3e-6** (user's caveat, confirmed).

## Interpretation
1. **The 1e-5 divergence is dominated by the int path** (weights + biases): removing the weight bifurcation (3e-6)
   cuts errors ~4× and turns the abrupt jump at ~326 into a gradual rise.
2. **The 3e-6 residual (19.7%) has two live contributors that this run cannot fully separate:**
   - **float drift** — the fp32 parameters (BN, fc) update by `w ± coeff` where `coeff` differs at a ulp (from the fp32
     tail's `g`), and the difference compounds: it is the only thing moving during the first ~300 ulp-only steps and it
     grows 3e-6 → 6e-5 → 4e-3 over the carry. Source.
   - **bias int path** — its `rint(mul·ratio)` updates diverge only once the drifted `g` differs enough (block-0 stays
     identical; blocks 1–4 differ by 4–12 units), after which a bias unit difference flips post-shift int8 activations
     (bias is added before the `>>16`) — a second, discrete amplifier. Consequence, then amplifier.
3. Reconciles the exp11 surrogate: a constant 3e-7 seed never bifurcates direct-int8; the real seed grows via float
   drift, and that is what crossed the weight threshold at step ~326 at 1e-5.

## Next control (not launched — user decision)
**exp12b: lr 3e-6 with the int32 bias updates disabled** (only fp32 params train; int8 weights already frozen).
Errors ≪ 4,254 → the bias amplifier matters; ≈ 4,254 → pure float drift. Implementation: drop the 5 `bias_rqsadd`
RQSPerturb nodes from the update set (exporter option) so the fixture is otherwise identical.
