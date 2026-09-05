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

## Three-way disagreement: Brevitas (A) vs true-int8 host (B-host) vs device (B-device)

_Added 2026-09-05 16:24 CEST. Two puzzles: (1) Brevitas A (88.89%) is "direct int8 too" —
why does it diverge MORE from B than B-device and B-host diverge from each other? (2) The
device update is direct int8 (not accumulated) and the per-step forward diff is only ~1e-7;
for that to bifurcate a weight it must land near a rounding threshold, which should be rare —
yet the great majority of losses differ. How?_

### Puzzle 2: "most losses differ" is two different phenomena, not one

Decoding the 10800 device +eps loss bit-words vs host-ref `loss_plus` and **categorizing**
each residual (not just "equal / not equal") splits it cleanly:

| category | count / 10800 | what it is |
|---|---|---|
| bit-exact | 165 (1.5%) | fp32 tail happened to land on the same bits |
| **ulp-only** (rel < 1e-5) | 1159 (10.7%) | **forward fp32 noise — weights identical, expf/logf/FMA differ** |
| medium (1e-5–1e-3) | 570 (5.3%) | transition |
| **LARGE** (rel ≥ 1e-3) | 8906 (82.5%) | **weight bifurcation — trajectories in different weight space** |

Per-step band (fraction bit-exact / ulp-only / LARGE):

| steps | bit-exact | ulp-only | LARGE |
|---|---|---|---|
| **0–1** (weights provably identical) | 0% | **100%** | **0%** |
| 0–10 | 17% | 75% | 2.5% |
| 10–100 | 12% | 83% | 0% |
| 100–500 | 7% | 52% | 16% |
| 500–1500 | 0% | 0.1% | **97%** |
| 1500–2700 | 0% | 0% | **99%** |

**Resolution.** "Non-bit-exact" and "threshold-crossing" are different events, and the
puzzle conflated them:

1. **Why do most losses differ at all?** Because two distinct fp32 forwards never bit-match.
   At step 0 the int8 weights are *provably identical* (single-step 0/16) yet **100% of the
   losses already differ** — all at ~1e-7 (ulp), none large. This ubiquitous ulp noise
   (expf/logf/FMA under `-ffast-math`) is what makes the overwhelming majority non-bit-exact.
   It has nothing to do with rounding thresholds or weight changes.
2. **Does the ~1e-7 land near a threshold "so often"?** No. LARGE (percent) differences —
   the actual weight bifurcations — are only **2.5% of losses in the first 100 steps**. The
   threshold crossing IS rare per step, exactly as intuition says. It only *looks* frequent
   later because **crossings are permanent and accumulate**: once one int8 weight flips it
   stays flipped, so the fraction of drifted weights climbs monotonically until (step 500+)
   the whole weight vector has separated and ~all losses read LARGE-different. The direct
   (non-accumulated) update does not accumulate the ulp *in one step*, but it does accumulate
   the *bifurcations across steps*. Ubiquity = forward ulp noise; growth = cumulative rare
   crossings. Two separate mechanisms.

### Puzzle 1: Brevitas A is a different computation, not "direct int8"

The premise "Brevitas was direct int8 as well" is the misconception. **Brevitas is
fake-quant**: fp32 values *snapped* to the quant grid (`round(x/s)·s`), the conv accumulated
in **fp32**, the rescale done by the **exact float scale** with a single float round at the
activation quantizer. **B is the true integer datapath**: `int8×int8→int32` exact
accumulation, rescale by a **dyadic fixed-point `mul >> shift`**, integer round-half-up. A
*simulates* int8 in float; B *is* int8.

**Decisive, training-free measurement — zero-shot, identical weights, no ZO at all:**

| actor | zero-shot b2 | trained (round-1) |
|---|---|---|
| **A** Brevitas fake-quant (fc-float) | **85.00%** | 88.89% |
| **B-host** true int8 `run_onnx_graph` | **83.33%** | 86.67% |
| **B-device** true int8 GVSoC | **83.33%** (= B-host) | 85.56% |

The **1.67-point A-vs-B gap exists with zero training** — so it is purely a *forward
algorithm* difference, not a trajectory difference. And **B-host = B-device at zero-shot**
(the integer forward is bit-exact across platforms; the fp32 tail is identical on a single
forward, proven by 0/16). So the three-way ordering is set before a single ZO step.

**Exactly where the A↔B drift lives.** Only the **5 conv requant layers** are quantized (fc
is float on both A and B), so the gap is entirely in those requants. The deployed graph
realizes each output channel's scale as `mul / 2^16` with **`mul` a small integer 60–456**
(measured from `network_zo_train.onnx`) — a per-channel scale granularity of `~1/mul ≈
0.2–1.6%`. It *cannot* represent Brevitas's exact float scale. That systematic per-channel
scale mismatch, plus integer round-half-up vs Brevitas's float round, shifts B's logits from
A's by ~1.4% in the **same direction** across the batch — hence **logit cos 0.986** (not
0.9999) and A sitting ~1.7 pt above B at zero-shot, ~2–3 pt above after training. It is a
*systematic* offset (Brevitas is a slightly optimistic proxy), not ulp noise.

### Why A diverges more — the hierarchy

| pair | what differs | magnitude | visible at zero-shot? |
|---|---|---|---|
| **A ↔ B** | the **computation** (float fake-quant vs integer fixed-point requant) | systematic ~1.4% logits / ~2 pt acc | **yes** (85.00 vs 83.33) |
| **B-host ↔ B-device** | the **platform** (numpy vs `-ffast-math`), *same* computation | ulp ~1e-7 forward; integer part **bit-exact** | no (both 83.33) |

A diverges more because **it is not the same computation as the device** — it is a float
approximation of quantization that rounds ~2 pt optimistically, and that gap is baked into
the forward, present before any training. B-host and B-device *are* the same computation, so
they agree bit-exactly on the integer forward and differ only in the fp32 tail — which merely
reshuffles the ZO random walk to nearly the same endpoint (86.67 vs 85.56), never a
systematic offset. Put simply: **A↔B is a model difference; B↔B is a rounding-of-the-last-bit
difference.**

## Files

- `analyze_device_vs_hostref_loss.py` — decodes the 10800 device loss bit-words and
  reproduces the residual-growth table (host-only, no container).
- `baked200_device_round1.log[.gz]` — device round-1 run with per-step `lp_bits` traces.
- `baked_200ep/outputs.npz` — host reference (`run_onnx_graph`) `loss_plus`/`loss_minus`/grad.
- device kernels (READ-ONLY refs): `Gemm.c` (FMA/unroll), `GlobalAveragePool.c` (seq accum),
  `SoftmaxCrossEntropyLossTemplate.py` (`expf`/`logf`); build flags in
  `TEST_SIRACUSA/build_master/compile_commands.json` (`-ffast-math -O3`).
