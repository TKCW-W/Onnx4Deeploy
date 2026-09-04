# exp10 — Deep research: why on-device QZO accuracy (83%) < Brevitas-sim (88.89%)

Goal (supervisor-directed): the accuracy drop is NOT accepted as a nature of quantized ZO —
direct-int8 QZO reaches good accuracy on the supervisor's benches. Find why OUR pipeline drops
it. Ensure (1) a faithful PyTorch+Brevitas reference, (2) a faithful on-device sim, (3)
bit-exactness AND float-ZO-like accuracy. Understand before implementing.

Actors:
- **A** = PyTorch + Brevitas fake-quant (fc-float, pooled@99.99) → the reference *accuracy* (88.89%)
- **B-host** = `run_onnx_graph` on the exported true-integer graph → reference *losses*, and
  (as of exp5) also an accuracy (83.89%)
- **B-device** = GVSoC compile of the same integer graph (82.95% measured); reproduces B-host
  to 1 window; loss drift vs B-host is fp32-chaos, not a compute difference (exp9 — settled).

---

## Iteration 1 (2026-09-05) — CORRECTS exp5's headline

exp5 concluded "the perturbation operator differs / no bug, just two rounding regimes." That
comparison was flawed: it compared A-at-A-state vs B-at-B-state, conflating the perturbation
operator with the forward. Corrected isolation (`isolate_forward_vs_perturb.py`):

1. **Perturbation operator is NOT the cause.** A's uniform per-channel step vs B's per-element
   `(z·mul+2^14)>>15` land at a different int8 code on only **33/14,880 = 0.2%** of conv weights.

2. **The FORWARD diverges under perturbation — this is the real effect.** Evaluating A's forward
   at B's EXACT +ε perturbed weights (identical state):

   | mb | A@B-state | B@B-state | diff |
   |---|---|---|---|
   | 0 | 1.261 | 1.119 | 0.14 |
   | 1 | 0.163 | 0.334 | 0.17 |
   | 2 | 0.114 | 0.048 | 0.066 |
   | 3 | 1.405 | 1.195 | 0.21 |

   vs unperturbed disagreement 0.006–0.039 (exp5 Control 1). **The A/B forward gap grows ~5–10×
   under a ±7-LSB perturbation** — exactly where ZO samples. That is what makes g_A=+19.3 vs
   g_B=−1.8.

**Reframing:** this is a forward-fidelity problem. A (fake-quant) and B (true integer) agree at
the operating point (cos 0.9994) but diverge at perturbed points. One of them is not a faithful
int8 forward. Since the supervisor gets good direct-int8 accuracy, a faithful int8 forward that
trains well exists — the task is to find which of A/B is wrong and where.

## Iteration 2 (2026-09-05) — ROOT CAUSE FOUND: RequantShift truncation corrupts the ZO gradient

Traced the requant semantics (shipped `Onnx4Deeploy_ZO` RequantShift == ours, inherited): the
kernel adds the rounding constant `2^(d-1)` ONLY when the `add` (bias) input is a compile-time
initializer; when `add` is a VARIABLE (our design, to make the conv bias ZO-perturbable) it
**truncates** (`>>d`, floor). Brevitas A rounds.

Decisive test (`test_requant_rounding.py`, and re-export with `QZO_FORCE_REQUANT_ROUND=1`),
step-0 g_proj with the SAME z:

| forward | g_proj[0] |
|---|---|
| B truncate (current) | **−1.81** (wrong sign) |
| B round (requant fixed) | **+17.06** |
| A Brevitas (same z) | **+19.32** |

**Rounding the requant flips B's gradient sign and aligns it with Brevitas.** The truncation —
forced by the variable-add bias design — is the accuracy bug. Full g-sequence also changes
throughout (truncate `[-1.81, 24.16, -24.85, 10.24, ...]` vs round `[17.06, 27.50, -17.21,
22.13, ...]`).

Mechanism: floor((k·mul + λ)/2^d) responds asymmetrically to the +ε vs −ε accumulator shift, so
the truncation bias does NOT cancel in L₊−L₋ and systematically corrupts g (here, sign-flip).
Invisible unperturbed (cos 0.9994); fatal to ZO, which reads only L₊−L₋ at perturbed points.

**Fix direction (device):** make the requant ROUND even with a variable `add` — add `2^(d-1)`
before the shift in the RQS path regardless of whether the bias is baked. This is a device
RequantShift-kernel change + the host reference to match (env hook `QZO_FORCE_REQUANT_ROUND`
added to `run_onnx_graph` as the diagnostic).

Confirmation in progress: full round-1 B-host with rounding -> batch-2 accuracy (expect recovery
toward ~88%). Then port the rounding to the device kernel + re-run bit-exactness.

### (superseded) earlier next-steps
- Trace the SHIPPED `Onnx4Deeploy_ZO` QZO reference path entry→output: how does IT compute the
  int8 forward + reference loss? Compare requant semantics (round vs truncate, fixed-point,
  per-channel) against our B and against Brevitas A.
- Localize the A/B forward divergence per layer (which requant/dequant boundary amplifies under
  perturbation).
- Hypothesis to test: B's RequantShift **truncates** (`>>S`) while Brevitas A **rounds** — a
  half-LSB bias per layer, invisible unperturbed (calibrated to match) but amplified when
  perturbation pushes activations across rounding boundaries.
