# exp11 — QZO device ↔ host-reference bit-exactness: plan

Started 2026-09-06 · Branch `feat/QZO` · Deliverable: a complete round-1 on-device QZO
simulation that is **bit-exact** to the host reference (every loss and every weight, diff=0).

## 1. Current status

- Single-step on-device QZO is bit-exact given identical weights (exp9/exp10, 0/16 diff).
- Full round-1 (2700 steps × 4 accum) on device reaches 85.56% (= host-ref accuracy) but the
  **trajectory diverges**: step-0 loss residual ~1e-7, growing to ~1e-1 by step 500+
  (exp10 `DIVERGENCE_NOTES.md`). The integer ops (Conv, RequantShift, RQSPerturb) are bit-exact.
- exp10 concluded the divergence was "by nature" (fp32 last-ulp under `-ffast-math`) and
  "nothing to align". **This plan reviews and corrects that conclusion.**

## 2. Root cause — reviewed

### 2a. What exp10 got right
The residual is fp32-only: the integer datapath is bit-exact, the noise vector `z` is bit-exact
(perturbation confirmed aligned below), and the growth mechanism (rare int8 flips that are
permanent and accumulate through the `round()` update) is correct.

### 2b. What exp10 got wrong — and why the user's "semantics mismatch" thesis is correct
exp10 attributed the residual to the *platform* (GAP9 FPU `fmadd`, compiler reassociation,
picolibc `expf`/`logf`) and treated the device build as immutable. Two errors:

1. **It missed that most of the fp32 mismatch is in our own host code, not the platform.**
   An op-by-op audit of `run_onnx_graph` (`onnx_node_implementations.py`) against the device
   kernels/templates finds six real *semantic* differences — different precision domain,
   different arithmetic operation, or different evaluation order — that no platform can be
   blamed for:

| # | op | host `run_onnx_graph` | device kernel / template | mismatch |
|---|---|---|---|---|
| 1 | **Quant** | `x.astype(float64) / scale` (fp64, **divide**), zp added **after** round | `scaled = x * (float32)(1.0/scale)` (fp32, **multiply** by fp64-computed reciprocal), zp added **before** round | domain + op + zp order → **feeds the int8 round** |
| 2 | **Dequant** | `((q - zp) * scale)` in **fp64** → cast fp32 | `((float32)q - (float32)zp) * (float32)scale` in **fp32** | domain (double rounding) |
| 3 | **BatchNormInternal** | `sc*(x-mn) / np.sqrt(vr+eps) + bs` (γ first, **divide** by sqrt) | `inv_std = 1/sqrtf(var+eps)`; `y = ((x-mean)*inv_std)*g + b` (reciprocal, γ last) | order + reciprocal-vs-divide (+ FMA on `*g + b`) |
| 4 | **GlobalAveragePool** | `np.mean` (pairwise sum, **divide** by HW) | sequential `sum += x[i]`; `sum * (1.0f/HW)` | order + reciprocal-vs-divide |
| 5 | **Gemm** (fc, M=1,N=32,O=9, transB) | BLAS `np.matmul` | 6-way unroll: `sum += p0+p1+p2+p3+p4+p5` (6-term left-assoc temp), tail loops | order (+ FMA) |
| 6 | **SoftmaxCrossEntropyLoss** | `np.exp/np.log`, pairwise `np.sum`, `np.mean` | sequential `sum += expf(..)`, `logf`, `total/batch` | libm + order |

   Confirmed **aligned** (no action): Perturb (host `w += ±eps` fp32; device `src + r*eps`
   with `r=±1.0f` — `r*eps` exact, single add, FMA-invariant), Relu/MaxPool (exact max),
   RequantShift/RQSPerturb (integer).

2. **It under-scoped the consequence.** Rows 1–3 sit *between* integer stages and feed the
   next block's int8 `Quant` round. A mismatch there flips int8 **activations**, not just
   the fc weights — so part of the "LARGE" divergence exp10 attributed to the fp32 tail is
   plausibly activation flips from host fp64-vs-fp32 Quant/Dequant. exp10's "integer datapath
   is bit-exact" was true of the integer ops but was never a proof that the fp32 glue between
   them was aligned. It is not.

3. **It treated `-ffast-math` as a wall.** It is a build flag on *our* vendored TrainDeeploy.
   With clang 15 (confirmed) we can scope `#pragma clang fp contract(off)` /
   `reassociate(off)` to the fp32 kernels so the device executes the C source order exactly.
   The only genuinely platform-bound piece is picolibc `expf`/`logf` (binary-only `libm.a`,
   no source) — and that is removable by owning the implementation on both sides.

**Corrected root cause:** the divergence is a set of *implementation-semantics mismatches*
between two hand-written realizations of the same math (rows 1–6), all alignable. It is not
by nature. The user's diagnosis stands.

### 2c. Verification bar
Bit-exactness must hold for the *carried* trajectory: all 10800 per-forward losses
(device `lp_bits` hex vs host `loss_plus`) and the final int8/fp32 weights, diff=0 over the
full round-1. Single-step is necessary but not sufficient (exp10 showed it passes while the
carry diverges).

## 3. Alignment policy — "the same implementation, and the most reliable + performing one"

Rather than nudging one side toward the other ad hoc, fix **one canonical fp32 semantics**
and implement it identically on both sides:

- **Precision domain: fp32 everywhere.** The device FPU is fp32-only (fp64 = soft-float,
  ~10–50× slower); fp32 is the standard, sufficient domain for int8 quantization. → host drops
  all `float64` in the quant glue. *Reliable* (no double rounding) and *performing*.
- **Reciprocal-multiply over divide** where the device already does it (Quant, BN inv_std,
  GAP `1/HW`): a multiply is cheaper than a divide on the device, and it is what the shipped
  kernels do. → host mirrors the reciprocal exactly (`np.float32(1.0/x)` computed the same
  way the device literal is produced).
- **Sequential / source-order reductions.** Deterministic and identical on both; the fp32
  tail (GAP over 10, fc 32×9, SCE over 9) is negligible cost vs the int8 convs, so no
  performance is lost by disabling FMA/reassociation *only there*.
- **Own the transcendentals.** One deterministic fp32 `expf`/`logf` (range reduction +
  polynomial, no FMA) in C on the device and op-for-op in numpy on the host, validated
  bit-exact by an input sweep and accurate to ≤2 ulp. Removes the only platform-bound piece.
- **Device change is scoped and gated** (pragmas / a compile definition on the fp32 kernels
  + SCE template), so the int8 fast path is untouched. Shipped originals kept commented
  alongside per the project rule.

## 4. Plan

| phase | work | exit criterion |
|---|---|---|
| **P0** | this Plan; `exp11/` layout | Plan.md committed |
| **P1 host mirror** | Rewrite `run_onnx_graph` Quant, Dequant, BatchNormInternal, GlobalAveragePool, Gemm(transB), SCE in np.float32 in the device's exact op order; shared `expf`/`logf` in numpy | host graph still yields 83.33% zero-shot / same argmax; per-op unit tests vs a C reference of each kernel (compiled strict on host) diff=0 |
| **P2 device strict** | `#pragma clang fp contract(off)`+`reassociate(off)` on BatchNorm.c, GlobalAveragePool.c, Gemm.c, SCE template; replace `expf`/`logf` calls with the shared implementation; verify the Quant scale literal is emitted with full repr | builds; single forward loss bit-exact to P1 host (diff=0) at step 0 for all 4 accum samples |
| **P3 single-step** | 2-step on-device run vs host | all losses + updated weights diff=0 |
| **P4 full round-1** | 2700-step device run (kill orphan gvsoc by PID first), decode all 10800 `lp_bits`, compare vs host `loss_plus`; compare final weights; device eval accuracy | **0 / 10800 loss mismatches, 0 weight mismatches**; accuracy reported |
| **P5** | Findings.md: accepted diffs with snippets, results, residual risks; last section = plan for item 2 (faithful PyTorch sim for retuning) | committed |

Risks / checks carried into P2–P4: (a) Mako must render the Quant/Dequant `${scale}` with
full `repr` (else the host cannot reproduce the fp32 literal) — verify on generated C;
(b) any remaining `float64` promotion in numpy scalar ops (guard with explicit `np.float32`);
(c) confirm the device training build takes the frozen-BN branch (`BN_FROZEN_STATS`);
(d) multi-core chunking of GAP/BN/Quant is per-element/per-channel, so core split does not
change reduction order — verify for Gemm (M=1 ⇒ single core).

## 5. Files (to be produced here)
`Plan.md` (this) · `Findings.md` · `host_mirror/` (aligned numpy ops + unit tests) ·
`shared_fp32_math.{c,py}` (expf/logf) · `device_round1.log[.gz]` · `verify_bitexact.py` ·
device/host loss + weight comparison artifacts.
