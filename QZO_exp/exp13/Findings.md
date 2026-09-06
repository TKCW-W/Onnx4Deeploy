# exp13 — Findings: train ONLY the fp32 parameters (conv weight + bias frozen), lr 3e-6

2026-09-07 01:05 CEST · Branch `feat/QZO` · Commands in `Plan.md`. Logs: `device_round1_3e6_freeze.log.gz`,
`export_3e6_freeze.log.gz`; params `device_weights_3e6_freeze.npz`, `baked_3e6_freeze_ref/`; 3-way `analysis.txt`.

## Setup
exp12 with the 10 `*_pmul` initializers zeroed in both graphs (`qzo_transform.freeze_conv_pmul`, exporter env
`QZO_FREEZE_CONV=1`): conv int8 weights and int32 biases neither perturb nor update on device or host. Verified: export
reports 10 zeroed initializers per graph; the export's graphs are byte-identical to the hand-frozen device fixture;
`inputs.npz`, node ids and the float-parameter noise are identical to exp12. Device run overlapped with the export;
errors = harness rule (|dev − ref| > 0.001 abs) recomputed from the raw `lp_bits`/`lm_bits` (validated on the 1e-5 log).

## Result (three device-level controls)
| run | int path | errors / 21,600 | first LARGE L+ / L− | final params device vs host |
|---|---|---|---|---|
| lr 1e-5 (exp10) | weights + biases train | **16,565 (76.7%)** | 5 / 326 | all classes differ |
| lr 3e-6 (exp12) | weights frozen, biases train | **4,254 (19.7%)** | 301 / 424 | int8 weights exact; biases 81/104 differ; fp32 ≤2.5e-4 |
| lr 3e-6 frozen (exp13) | **weights + biases frozen** | **5,838 (27.0%)** | 181 / 204 | **10/10 int tensors bit-exact**; 12 fp32 tensors differ, max 1.7e-4 … 6.8e-4 |
exp13 per band (L+): 0–300 99.5% ulp-only; 300–600 4% LARGE; 600–1200 23%; 1200–2700 81% (median rel 1e-2).

## Interpretation
1. **The bias int path is not an amplifier.** Removing it did not reduce the disagreement (it rose; a different chaotic
   realization — the robust statement is that the residual does not need the bias path).
2. **The entire 3e-6 residual is the float path.** With every int parameter bit-exact end-to-end, device and host still
   disagree on 27% of losses. Mechanism: fp32 params (fc, BN γ/β) receive `w ± coeff` where `coeff = f32(−lr·g_proj)`
   differs at a ulp because the fp32 forward tail differs at a ulp (Gemm/BN FMA, picolibc `expf`/`logf`); the difference
   compounds (2e-4–7e-4 by the end); BN drift crosses int8 activation-Quant boundaries → LARGE single-forward
   differences → larger `g` differences → faster drift. The activation Quant is the amplifier inside the float path —
   which is why float ZO (same drift, no Quant) stays within ~1e-4 and QZO does not.
3. **At 1e-5 the int8 weight path adds ×4** (bifurcation at step ~326).

## Consequence for the deliverable (device ↔ host bit-exact carry)
L1/L2 made every host op identical to the device; the update path is verified identical (source + compiled); exp13
shows the int path can be made exactly identical. The only remaining source of difference is the fp32 tail seed.
Everything downstream of it is deterministic on both sides, so removing the seed (L3: `-fno-fast-math -ffp-contract=off`
scoped to the fp32 kernels/graphs — the only combination measured to give fused=0; L4: one deterministic `expf`/`logf` on
both sides) would make `g`, `coeff`, the fp32 parameters and therefore the whole 2700-step carry bit-exact (diff = 0),
not merely within tolerance. Conversely, no change confined to the int update path (freezing, master weights) can reach
float-ZO tolerance while the seed remains: exp13 is "fp32-only training on device" and still diverges to 27%.
Decision pending with the user.

## Micro-trace (2026-09-07) — see `micro/FINDINGS.md`
Layer-by-layer device-vs-host bit-level probes of the step-0 +eps forward: **every integer stage bit-exact** (all five
conv+requant outputs 0 differing elements); **the first difference appears at BN-0** (25% of elements, mostly 1 ulp,
max 4) and is re-created by every BatchNormInternal (25–46%), absorbed by the following Quant, and reaches the loss only
through BN-4 → GAP → fc (logits 7/9 at 1–2 ulp → loss 5 ulp). The int path does not participate. Causal one-kernel
confirmation (strict-fp BatchNorm.c, re-run probe 1 → expect 0) proposed, awaiting go/no-go.
