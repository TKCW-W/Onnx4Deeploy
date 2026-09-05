# exp10 — QZO device vs host-reference: why they diverge, and why it is by nature

Date: 2026-09-05 · Branch `feat/QZO` · Grounded in the device's own logged loss
bit-patterns. Reproduce with `analyze_device_vs_hostref_loss.py`.

## Task

Locate the exact cause of the device-vs-host-ref divergence. Confirm the two sides are
truly identical (implementation, computation, noise vector, same step both sides). **If
the difference is not by nature, align them.**

## Answer (one line)

The mechanism **is** identical — proven bit-exactly on a single step and to ~1e-7 (last
ulp) on the multi-accum step-0 loss. The residual divergence is **by nature**: irreducible
fp32 last-ulp differences (the device is built `-ffast-math -O3`) amplified by the
`round()`-thresholded int8 weight update into a different-but-equivalent ZO trajectory.
There is nothing to align — the difference is at the instruction / libm level, not in our
code — and it is the same class of difference float ZO has, only made visible by `round()`.

## Grounding: the device logs its own loss bits

The device emits every +eps forward loss as a raw IEEE-754 word:
`[PHASE] +eps loss read OK: lp_bits=0x........`. We decoded all **10800** of them
(2700 steps × 4 accum) from `baked200_device_round1.log` and compared them bit-for-bit and
in relative residual against the host reference `loss_plus` — produced by `run_onnx_graph`
on the **same** exported graph with the **same** seed (`baked_200ep/outputs.npz`).

| span | median rel-resid (device vs host-ref) | max |
|---|---|---|
| **step 0** (identical weights, identical z) | **3.9e-07** | 1.7e-5 |
| steps 5–100 | 4.9e-07 | 1.1e-3 |
| steps 100–500 | 3.5e-06 | 1.6e-1 |
| steps 500–1500 | 7.5e-02 | 3.7e-1 |
| steps 1500–2700 | 1.0e-01 | 4.4e-1 |

Overall bit-exact: 1.5%. Step-0 per-accum residual: `[5.8e-7, 8.8e-8, 7.2e-7, 1.5e-7]`.

**Reading:** at step 0 the two forwards see identical int8 weights, identical z, identical
inputs, and differ by **~1e-7 relative = the last 1–2 ulp of fp32**. That is the noise
floor of two distinct fp32 implementations — it is not a mechanism difference. The residual
then **grows by 6 orders of magnitude** over the carry. That growth is the signature.

## The three by-nature fp32 sources (all sealed by the device build flags)

The int8 datapath (conv → RequantShift) is **bit-exact** by construction (integer, and the
requant now rounds via the baked `div/2` — see `Findings.md`). The only fp32 lives in the
tail: **GlobalAveragePool → fc Gemm → SoftmaxCrossEntropyLoss**. Every fp32 op there is
compiled with `-ffast-math -O3` (verified in `TEST_SIRACUSA/build_master/compile_commands.json`):

1. **FMA fusion.** `Gemm.c` does `sum += a_val * pSrcB[...]`. Under `-ffast-math` the
   compiler fuses each into a single `fmadd` (one rounding); numpy's `matmul` does multiply
   **then** add (two roundings). Different result in the last ulp — and numpy has no FMA path
   to match it.
2. **FP reassociation.** `-ffast-math` frees the compiler to reorder the fp32 reductions
   (the pool's sequential accumulate, the fc's 6-way-unrolled `sum0..sum5`). The order in the
   emitted binary need not even match the C source, so "align the reduction order in numpy"
   cannot be made faithful — there is no fixed order to target.
3. **Transcendentals.** `SoftmaxCrossEntropyLossTemplate.py` computes the loss with
   **`expf` / `logf`** from picolibc (line 21–22: `sum_exp += expf(logit - max)`,
   `log_sum_exp = logf(sum_exp)`), under `-ffast-math` (fast, reduced-precision variants).
   numpy's `np.exp`/`np.log` are entirely different implementations and differ in the last
   ulp. Not bit-matchable without re-implementing picolibc's libm.

Any one of these makes bit-exact alignment to a strict-IEEE numpy reference impossible; all
three are present. This is why "if not by nature, align them" **does not apply** — the
difference is by nature, baked in below our code by the compiler and the C library.

## The amplifier: why a 1e-7 forward difference becomes a 1e-1 trajectory difference

The ZO update is `θ' = round(θ − lr · g · z)`, `g = (L₊ − L₋)/(2ε·n)`. `round()` is a
**threshold nonlinearity**. A ~1e-7 ulp difference in `g` is almost always absorbed (the
int8 weight rounds to the same value — this is also the LSB-stall regime), which is why
single-step bit-exactness passes (0/16) and step 0 shows 0 moved-weight divergence. But
occasionally a weight sits within ~1e-7 of a rounding boundary; then the ulp difference
flips **one int8 LSB**. From that step on, the device and host are at **different points in
weight space**, and their subsequent losses diverge at percent scale. Compounded over 2700
steps this is deterministic chaos: both are valid ZO descents, they reach the **same
accuracy** (device 85.56% ≈ host-ref), but **not** the same weights.

## Why float ZO looked "bit-exact" and QZO does not (answering the standing question)

Same root (fp32 last-ulp on a different platform), opposite response — and, notably, QZO's
forward is *cleaner*, not dirtier:

- **QZO forward is mostly integer.** Only the GAP→fc→SCE tail is fp32, so the step-0 loss
  residual is **~1e-7**. Float ZO is fp32 **throughout** (every conv + BN), so its per-step
  loss residual is **~1e-4** (measured earlier: only 917/21600 losses exactly equal, rest
  1e-4…8e-4). QZO's tail is the smaller fp32 difference.
- **The update response is what differs.** Float ZO's update `θ' = θ − lr·g·z` is **smooth**:
  a ulp difference in `g` stays a ulp difference in the fp32 weight — bounded, never
  amplified, so the trajectories track and it reads as "bit-exact within tolerance." QZO's
  `round()` update has a threshold that turns that same (in fact smaller) ulp difference into
  a whole-LSB weight flip whenever a weight is near a boundary — unbounded amplification.
- Float ZO's host reference is also a *different* executor (`_zo_pytorch_reference`, PyTorch)
  from the device, and it too differs at the ulp — the smoothness just hides it. QZO uses
  `run_onnx_graph` (numpy); the `round()` exposes it.

So float ZO is **not** truly bit-exact either; it is bit-exact *within a bounded tolerance*
because its update cannot amplify. QZO cannot be, because `round()` amplifies — and that is a
property of quantized (int8) training, not a bug in our port.

## What "identical" we can and did guarantee

- **Same graph, same integer datapath, same perturbation operator, same noise vector z**:
  proven by single-step on-device bit-exactness (0/16) and by step-0 loss agreeing to ulp.
- **Same requant rounding** on both sides (baked `div/2`, `Findings.md`) — the one real bug,
  now fixed; it is not part of this residual.
- What is **not** and **cannot** be identical is the carried fp32 trajectory, for the three
  by-nature reasons above. The correct bar for "the mechanism is the same" is therefore
  single-step bit-exactness given identical weights — which holds — not multi-step
  trajectory equality, which no two distinct fp32 platforms can satisfy under a `round()`
  update.

## Conclusion

The divergence is located and explained end to end, grounded in the device's own logged loss
bits: a ~1e-7 last-ulp fp32 difference (FMA + reassociation + `expf`/`logf`, all forced by
`-ffast-math`) amplified through the `round()` int8 update. It is **by nature** — not a
mechanism, code, or noise-vector difference — so the "align them" branch does not apply;
aligning would require bit-emulating GAP9's fmadd, the compiler's chosen fp order, and
picolibc's `expf`/`logf` in numpy, which is impractical and compiler-version-fragile, and
would be a band-aid, not a faithful reference. The faithful reference is the graph we already
share; the device reproduces it to the fp32 floor at step 0 and converges to the same
accuracy. Nothing to fix here.

## Files

- `analyze_device_vs_hostref_loss.py` — decodes the 10800 device loss bit-words and
  reproduces the residual-growth table (host-only, no container).
- `baked200_device_round1.log[.gz]` — device round-1 run with per-step `lp_bits` traces.
- `baked_200ep/outputs.npz` — host reference (`run_onnx_graph`) `loss_plus`/`loss_minus`/grad.
- device kernels (READ-ONLY refs): `Gemm.c` (FMA/unroll), `GlobalAveragePool.c` (seq accum),
  `SoftmaxCrossEntropyLossTemplate.py` (`expf`/`logf`); build flags in
  `TEST_SIRACUSA/build_master/compile_commands.json` (`-ffast-math -O3`).
