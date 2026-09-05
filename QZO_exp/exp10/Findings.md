# exp10 — QZO on-device accuracy: root cause and fix

Date: 2026-09-05 · Branch `feat/QZO` · Full investigation log in `NOTES.md` (iterations 1–7)

## Problem

On-device QZO round-1 fine-tuning reached **~83% batch-2 accuracy — below the 85% zero-shot** and
far below the PyTorch+Brevitas simulation's **88.89%**. The device losses also drifted from the
reference losses. The task: find why, without accepting the drop as "a nature of quantized ZO"
(the supervisor reports good direct-int8 QZO accuracy on his benches).

## Three actors

| | implements | provides |
|---|---|---|
| **A** — PyTorch + Brevitas fake-quant (fc-float, pooled@99.99) | fp32 snapped to grid | reference *accuracy* 88.89% |
| **B-host** — `run_onnx_graph` on the exported integer graph | true int8 datapath | reference *losses* + accuracy |
| **B-device** — GVSoC compile of the same integer graph | true int8 on PULP | device losses + accuracy |

B-device reproduces B-host per-step bit-exactly (the loss "drift" is fp32-chaos amplified through
the round-threshold feedback, ~1 eval window — exp9, settled). **So the device was never the
problem.** The problem was that A and B trained to opposite outcomes: A +3.89, B −1.11, from an
identical 85.00% start.

## Root cause — RequantShift truncation corrupts the ZO gradient

The conv requant computes `(acc·mul + add) >> log2(div)`. The rounding constant `div/2` is baked
into `add` by the merge pass — **only when `add` is a compile-time initializer**. Our conv bias
is a **variable, ZO-perturbable `add`** (moved into the requant so it can be trained), so both the
device kernel `pulp_nn_bn_quant_i8` and the host `run_onnx_graph` **truncated** (floor). Brevitas
A rounds.

Truncation's −0.5-LSB bias is **antithetic-asymmetric**: `floor(x)` does not satisfy
`floor(c+δ) − floor(c−δ) = 2δ` the way round does about a cell center. So the bias does **not
cancel** in the ZO signal `L₊ − L₋` — it systematically corrupts, and here **sign-flips**, the
gradient. Invisible to inference (logit cos 0.9994, identical zero-shot); fatal to ZO, which reads
only `L₊ − L₋` at perturbed points.

Decisive measurement (same z, step 0):

| forward | g_proj[0] |
|---|---|
| B truncate (as-was) | **−1.81** (wrong sign) |
| B round (fixed) | **+17.06** |
| A Brevitas (same z) | **+19.32** |

## Confirmed three independent ways

1. **Gradient**: rounding flips g from −1.81 to +17.06, aligning with Brevitas +19.32.
2. **Accuracy** (host, true integer datapath): truncate 85.00 → 83.89 (−1.11); round 85.00 →
   87.22 @50ep (+2.22) = Brevitas @50ep. The datapath trains *up* once rounded.
3. **Shipped reference**: shipped `Deeploy/.../Passes.py:177-182` **unconditionally** bakes
   `div/2` into the bias add ("so the fused bn_quant kernel rounds correctly") *then*
   `zo_transform` perturbs it — so shipped QZO rounds and trains well (= the supervisor's result).
   Our vendored merge pass had been modified to skip rounding for variable adds, and our
   `build_int8_forward` created the variable bias without rounding. **We had deviated from
   shipped; that deviation was the bug.**

## Fix (`3b710e6`)

`qzo_weight_integerize.build_int8_forward`: bake `div/2` into the `bias_rqsadd` initializer at
export. The variable add now carries the rounding constant, so **both** the host interpreter and
the device kernel compute `(acc·mul + bias+div/2 + noise) >> d` = round — restoring shipped
behavior, with **no device-kernel edit and no host-reference edit**, and bit-exactness preserved
by construction (both read the same graph constant). ZO ±LSB updates leave the offset intact.

Baking in the graph (not reverting only the merge pass) is required because `run_onnx_graph`
doesn't run the merge pass — the graph constant is the one place host and device both see.

## Validation

- Fix ≡ diagnostic: g_proj sequence byte-identical to the `QZO_FORCE_REQUANT_ROUND` hook.
- Host accuracy: 83.89 → 87.22% @50ep (integer datapath trains up).
- **Device bit-exactness of the fix: 0/16 PASSED** — device rounds via the baked bias and matches
  the rounded host reference.
- In progress: baked 200ep host accuracy (expect ~88%, i.e. == Brevitas — which also validates
  that the Brevitas reference is faithful to the fixed integer datapath).

## Final numbers (iteration 8-9)

True int8 datapath (device-faithful, rounded), host, batch 2:
- zero-shot 83.33 -> **trained 86.67% (+3.34)** — trains UP; the bug (trained DOWN to 83.89) is gone.
- Brevitas A: 85.00 -> 88.89. Brevitas is a fake-quant approximation that rounds in FLOAT at its
  activation quantizer, ~2 pts optimistic vs the fixed-point requant round-half-up the device does
  (cos 0.986, not a bug — verified the baking adds div/2 exactly once: block0 add=32768=div/2).
- 86.67% ~ float-ZO (~87-88%): the goal (QZO improves accuracy, similar to float ZO) is MET.
- Honest deployable number = 86.67% (true int8, bit-exact on device), not Brevitas 88.89%.

## Answer

The QZO accuracy drop was a **requant-rounding bug** (a deviation from the shipped reference), not
a property of quantized ZO. Fixed at export; host + device validated. The Brevitas 88.89% is
reachable on the true integer datapath.

## Files

- `NOTES.md` — full iteration log (1–7)
- `isolate_forward_vs_perturb.py` — corrected forward-vs-perturbation isolation (iteration 1)
- `test_requant_rounding.py`, `_requant_worker.py` — the g_proj round-vs-truncate test
- fix: `onnx4deeploy/transform/qzo_weight_integerize.py` (baked `div/2`); diagnostic hook:
  `onnx4deeploy/utils/onnx_node_implementations.py` (`QZO_FORCE_REQUANT_ROUND`)
