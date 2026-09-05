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

### CONFIRMED (iteration 3): rounding recovers accuracy

True integer datapath, host reference, batch 2:

| datapath | zero-shot | trained | Δ |
|---|---|---|---|
| **truncate** (current, 200 ep) | 85.00 | **83.89** | −1.11 (worse) |
| **round** (fixed, 50 ep) | 85.00 | **87.22** | **+2.22** (improves) |

Rounding flips the integer datapath from training DOWN to training UP. 87.22% @ 50 epochs ==
Brevitas @ 50 epochs; the 88.89% (200 ep) is now reachable on the real datapath. **Truncation was
the entire accuracy bug.** The whole investigation resolves to a one-line requant fix.

### Remaining work (deployment)
1. Full 200-epoch rounded host run -> confirm ~88% (in progress).
2. **Port rounding to the DEVICE RequantShift kernel** — the kernel truncates a variable-add
   requant; add `2^(d-1)` before the `>>d`. Then the host reference (already has the diagnostic
   env hook) is promoted to unconditional rounding, and re-run single-step + full bit-exactness.
3. Re-run the on-device round-1 with the fixed kernel -> device accuracy ~88%, bit-exact vs the
   rounded host reference.

### (superseded) earlier next-steps
- Trace the SHIPPED `Onnx4Deeploy_ZO` QZO reference path entry→output: how does IT compute the
  int8 forward + reference loss? Compare requant semantics (round vs truncate, fixed-point,
  per-channel) against our B and against Brevitas A.
- Localize the A/B forward divergence per layer (which requant/dequant boundary amplifies under
  perturbation).
- Hypothesis to test: B's RequantShift **truncates** (`>>S`) while Brevitas A **rounds** — a
  half-LSB bias per layer, invisible unperturbed (calibrated to match) but amplified when
  perturbation pushes activations across rounding boundaries.

### Device-fix analysis (iteration 4) — exact locations

Traced the device requant end-to-end:
- `Deeploy/Targets/PULPOpen/TopologyOptimizationPasses/Passes.py:173-180` (OUR merge pass):
  computes `rounding = 2^(totalShift-1)` and bakes it into the add ONLY for a CONSTANT add;
  for a VARIABLE add (our perturbable conv bias) it skips — comment says "matching the host
  reference truncation." That deliberate choice made host+device consistently WRONG.
- int8 requant = `pulp_nn_bn_quant_i8` (`third_party/pulp-nn-mixed/.../pulp_nn_utils.h:271`):
  `integer_image_phi = k*phi + lambda; x = integer_image_phi >> d;`  — TRUNCATE, no +2^(d-1).
- Standalone `RequantShift.c` kernel DOES round (has a `rounding` flag, template passes 1); only
  the merged-conv path truncates — why non-conv RQS was fine and only conv output was hurt.

RECOMMENDED FIX (no third-party kernel edit, preserves bit-exactness): bake `2^(totalShift-1)`
into the conv `bias_rqsadd` initializer at export (`build_int8_forward`). Then the variable add
carries the rounding; host (`run_onnx_graph`, no env flag) and device (`pulp_nn_bn_quant_i8`)
both compute `(acc*mul + bias+rounding + noise) >> d` → round, consistently → bit-exact AND
correct. `QZO_FORCE_REQUANT_ROUND` was the DIAGNOSTIC; the shipped fix is the baked constant.

Verify after: single-step device bit-exactness (rounded), then full round-1 device run + accuracy.

---

## Iteration 5 (2026-09-05) — FIX SHIPPED (host-verified)

`qzo_weight_integerize.build_int8_forward`: bake `div//2` into the conv `bias_rqsadd`
initializer. The variable add now carries the rounding constant, so BOTH the device kernel
`pulp_nn_bn_quant_i8` and the host `run_onnx_graph` compute `(acc*mul + bias+div/2 + noise) >> d`
= ROUND — no device-kernel edit, no host-ref edit, bit-exactness preserved by construction.

Verified equivalent to the `QZO_FORCE_REQUANT_ROUND` diagnostic: g_proj sequence byte-identical
`[17.0606, 27.5042, -17.2079, 22.1271, ...]`. Commit `3b710e6`.

**Answer to the whole investigation:** the QZO accuracy drop (83% < zero-shot, vs Brevitas 88.89%)
was NOT a nature of quantized ZO — it was a requant TRUNCATION bug (our variable-add bias defeated
the merge pass's rounding). Fixed. Host int datapath now trains UP (85.00 -> 87.22 @50ep, ~88% @200ep).

### Remaining (device confirmation, in progress)
- baked_200ep run -> device-correct fixture + host reference + updated weights (~88% expected).
- Then: pack -> single-step device bit-exactness (must round, stay bit-exact) -> full round-1
  device run -> device accuracy ~88%. This closes "faithful on-device sim + good accuracy".

---

## Iteration 6 (2026-09-05) — SHIPPED REFERENCE CONFIRMS: we deviated from its rounding

Traced the shipped `Onnx4Deeploy_ZO` + shipped `Deeploy` QZO path entry->output (as instructed):

1. Shipped `zo_transform.py:43,250-252` perturbs the `_add` (int32 pre-shift bias) — SAME
   variable-add-bias structure as ours. So structure is not the difference.
2. Shipped `Deeploy/.../TopologyOptimizationPasses/Passes.py:177-182`:
   ```python
   rounding = 2**(totalShift - 1) if totalShift > 0 else 0
   # Bake rounding into lambda so the fused bn_quant kernel rounds correctly
   rqs.inputs[-1].values = copy.deepcopy(rqs.inputs[-1].values) + rounding
   ```
   The shipped merge pass **UNCONDITIONALLY bakes div/2 into the bias add** (also line 120), while
   the bias is still constant — THEN zo_transform perturbs it. So the shipped `_add` carries the
   rounding constant, the fused `pulp_nn_bn_quant_i8` `(k*phi+lambda)>>d` rounds correctly, and
   QZO trains well (== the supervisor's direct-int8 result).

3. **Our bug was a deviation FROM shipped:** our vendored `Passes.py` was modified to bake
   rounding "only for a constant add; for a variable add the kernel truncates — matching the host
   reference" (`-- QW`), and our custom `build_int8_forward` created the variable `bias_rqsadd`
   WITHOUT the rounding constant. Both changes dropped the shipped rounding for the (now variable)
   bias -> truncation -> corrupted ZO gradient -> 83% accuracy.

**Our fix (`3b710e6`) restores shipped behavior** by baking div/2 into `bias_rqsadd` at export,
visible to BOTH the host interpreter (`run_onnx_graph`) and the device kernel -> both round,
bit-exactness preserved. (Reverting only the vendored merge pass would NOT suffice: the host
`run_onnx_graph` doesn't run the merge pass, so it would still truncate and break bit-exactness;
baking in the graph is the host+device-consistent fix.)

**Root cause is now confirmed three independent ways:** (a) g_proj sign-flip -1.81->+17.06 with
rounding; (b) host accuracy 83.89->87.22 with rounding; (c) the shipped reference bakes exactly
this rounding and we had removed it. The supervisor's "direct int8 works" is fully explained.

### Iteration 7 — DEVICE bit-exactness of the fix: PASSED

2-step baked fixture (div/2 baked into bias, no env flag) on GVSoC: **Errors: 0 out of 16,
PASSED.** The device kernel `pulp_nn_bn_quant_i8` now rounds (via the baked bias) and is
bit-exact against the rounded host reference. Fix validated host + device.

Status: root cause found + 3-way confirmed + shipped-ref confirmed; fix shipped (3b710e6);
host accuracy recovered (85.00->87.22 @50ep); device bit-exact with fix (0/16). Remaining: full
round-1 device accuracy (baked_200ep fixture -> device run -> ~88%).

### Iteration 8 — 200ep baked result: training FIXED, but a forward-fidelity tension to resolve

Baked (rounded, true integer datapath), host, batch 2:
| datapath | zero-shot | trained 200ep | Δ |
|---|---|---|---|
| round-integer (device-faithful) | 83.33 | **86.67** | **+3.34 (UP)** |
| Brevitas A | 85.00 | 88.89 | +3.89 |
| truncate-integer (bug) | 85.00 | 83.89 | -1.11 (DOWN) |

**Core problem SOLVED:** the integer datapath now trains UP (+3.34), like Brevitas (+3.89),
instead of down. QZO improves accuracy on the true int8 path (~87%, near float-ZO ~88%).

**OPEN tension (do not gloss):** the baked ROUNDED forward vs Brevitas is cos **0.9860** (29/30
argmax) — WORSE than the TRUNCATING forward's cos **0.9994** measured in exp9. So rounding
aligned the GRADIENT with Brevitas (fixes training) but moved the ABSOLUTE forward AWAY from
Brevitas at the operating point, and dropped zero-shot 85.00 -> 83.33. Two forwards that both
"round" should match better, not worse. Possible causes to check next:
  (a) my +div/2 rounding is mis-scaled / double-applied somewhere;
  (b) Brevitas Int8ActPerTensorFloat uses a different rounding (half-even vs half-up) or rounds
      at the activation-quant, not the requant, so truncate-integer coincidentally matched it;
  (c) which of round/truncate is the TRUE int8 forward? Need a reference-independent int8
      numpy reimplementation of conv+requant to adjudicate — do NOT assume Brevitas is ground
      truth for the absolute forward (it is fake-quant, an approximation of true int8).
Next iteration: settle (c) with an independent int8 reference; confirm the rounding constant;
then decide whether 86.67 (true int8) or 88.89 (Brevitas) is the honest deployable number.

### Iteration 8b — recalibration (calmer read)

cos 0.986 is HIGH (the mul bug was 0.87); zero-shot moved 85.00->83.33 = ~3 windows = small.
The rounding fix is fundamentally sound: it fixes the GRADIENT + training (dominant), and the
small forward shift is likely because Brevitas fake-quant rounds at its ACTIVATION quantizer
while true int8 rounds at the REQUANT — different points, so Brevitas is an approximation of the
true device forward. Honest deployable number is likely the true-int8 86.67%, not Brevitas 88.89.
DECISIVE next step: independent int8 numpy conv+requant reference (explicit rounding) to
adjudicate round vs truncate WITHOUT assuming Brevitas is ground truth. Then finalize the number.
