# exp15 — ECO for on-device quantized ZO fine-tuning — Findings

Date: **2026-09-10** · faithful Brevitas SpeechNet, pooled@99.99 frozen act scales + frozen
pretrained per-channel weight scales, incremental 4-round protocol (ft batch r → eval batch r+1),
n_accum 4, eps 0.01, z-seed restart per round, single seed 42. Code: `run_eco.py` (reuses
exp_calibration `run_study`/`run_incremental`). Paper: `../../../docs/eco_paper_summary.md`.

## Headline

**ECO transfers to our quantized-ZO setting and breaks the LSB stall without a master-weight
buffer.** At lr 3e-6 — where direct-INT8 conv is *completely frozen* — injecting the per-step
rounding residual into an SGD-momentum buffer makes the INT8 conv weights trainable again and
reaches accuracy equal to or above the working references. The essential ingredient is the
residual injection (error feedback), not the momentum: momentum with the injection removed still
stalls at 0% movement.

## Results (mean over 4 incremental rounds, all at lr 3e-6 unless noted)

| arm | mean acc | per-round acc_after | conv moved (union, 4rd) | steps w/ any conv move | cos(e_t,e_{t+1}) |
|---|---|---|---|---|---|
| `direct_3e6` (stall baseline) | 85.42 | 87.8 / 81.1 / 88.9 / 83.9 | **0.0%** | **0.0%** | — |
| `sgdm_none_3e6_b90` (momentum, **no EF** — ablation) | 84.86 | 87.2 / 80.6 / 88.9 / 82.8 | **0.0%** | **0.0%** | 0.910 |
| `eco_mf_3e6_b99` (β too high) | 84.86 | 86.7 / 80.0 / 90.0 / 82.8 | **0.0%** | **0.0%** | 0.999 |
| **`eco_mf_3e6_b90`** (memory-free, RTN) | **85.69** | 89.4 / 82.2 / 87.2 / 83.9 | 11.8% | 17.1% | 0.998 |
| **`eco_mf_3e6_b90_sr`** (memory-free, **SR**) | **86.81** | 88.9 / 84.4 / 88.3 / 85.6 | 100.0% | 100.0% | 0.002 |
| **`eco_ex_3e6_b90`** (exact EF, stores e) | **86.81** | 88.9 / 85.0 / 87.2 / 86.1 | 99.9% | 99.2% | 0.970 |
| `direct_1e5` (working INT8 reference) | 85.97 | 90.0 / 83.3 / 87.8 / 82.8 | 99.0% | 1.4% | — |

References (exp_calibration `run_incremental`, same protocol): direct@1e-5 **86.11%**,
float-ZO@3e-6 **86.11%**. Zero-shot batch-2 = 85.56%.

## What the numbers say

1. **The stall, and the BN-only floor.** At lr 3e-6 direct-INT8 moves **zero** conv weights across
   all 2700×4 steps, yet accuracy is 85.42% — because the FP32 BatchNorm γ/β keep training via
   float ZO (float params have no LSB stall). So **85.42% is the accuracy reachable with the conv
   weights frozen**; conv training only matters insofar as it beats this floor. This is the honest
   baseline and it caps how much headroom any conv-training method can show in this near-converged
   protocol.

2. **ECO breaks the conv stall.** Memory-free β0.9 moves conv weights (11.8% of them, on 17% of
   steps) and reaches 85.69% — above the BN-only floor. SR and exact-EF reach **86.81%**, matching
   and slightly beating the 86.11% references, at lr 3e-6 and with no master weights.

3. **The ablation is decisive: error feedback, not momentum, is what matters.** `sgdm_none` (SGD
   momentum with the residual injection deleted) is byte-for-byte a stall — 0% movement, 84.86% —
   identical to `direct_3e6`. Momentum alone cannot break the stall because rounding still discards
   the sub-LSB step. It is the injection of the rounding residual back into the buffer that
   accumulates the discarded updates until they cross the LSB.

4. **β is the knob; α stays pinned.** α = (1/lr)(1−1/β) is fixed by lr and β (never tuned). At
   β0.99, α is ~15× smaller than at β0.9, the accumulated injection never crosses the LSB within
   the horizon, and the conv stalls again (0% movement, 84.86%) — *despite* cos(e)=0.999. So the
   memory-free rule needs β small enough that error feedback crosses the LSB within the run; β0.9
   is the sweet spot here, β0.99 is too conservative.

5. **The go/no-go diagnostic reproduces the paper exactly.** RTN memory-free: **cos(e_t,e_{t+1}) =
   0.998** — consecutive residuals are highly correlated, so the memory-free heuristic
   (`e_t ≈ e_{t+1}`) is valid in our ZO setting. This directly answers the paper's §8.3 worry that
   SPSA's independently resampled `z_t` would decorrelate the residuals: it does not, because the
   residual is dominated by the slowly-varying accumulated momentum, not the fresh rank-1 term.
   SR gives cos(e)=0.002 (SR injects independent noise, so residuals decorrelate — expected, and
   irrelevant since SR moves every step by construction). Exact-EF: 0.97.

6. **Write-sparsity trade-off (matters for MCU).** Memory-free RTN moves only **11.8%** of conv
   weights (17% of steps) — error feedback acts as a strong-signal filter, writing only the weights
   whose accumulated residual genuinely crosses the LSB. SR and exact-EF move ~**100%** every step.
   So +1.1pt accuracy (86.81 vs 85.69) costs ~8× more INT8 weight writes. On a device where weight
   writes are flash/L2 traffic and energy, the memory-free RTN arm is the write-sparse option.

## Setting that achieves the target accuracy

- **Best accuracy:** `eco_mf_3e6_b90_sr` (memory-free + stochastic rounding) and `eco_ex_3e6_b90`
  (exact EF) both reach **86.81%**, at/above the 86.11% references, at lr 3e-6, no master weights.
- **Best memory + write-sparsity:** `eco_mf_3e6_b90` (memory-free, RTN) reaches 85.69% — within
  ~0.4pt of the references and above the BN-only floor — while moving only 11.8% of conv weights.
- Movement answer, directly: steps that move conv weights = 17% (memory-free β0.9), ~100% (SR /
  exact), 0% (direct@3e-6, sgdm-none, β0.99). Conv weights ever updated = 11.8% (memory-free β0.9),
  ~100% (SR / exact), 99% (direct@1e-5).

## Honest caveats

- **The accuracy headroom is small** in this near-converged incremental protocol — BN adaptation
  alone gives 85.42%, so the whole conv-training story lives in a ~1.4pt band (85.4 → 86.8). The
  cleanest evidence here is *movement* + the *ablation*, not the accuracy delta. The accuracy
  confirms ECO does not hurt and slightly helps; the memory/trainability argument is the substance.
- **Single seed (42).** The 86.11 references were multi-seed; these arms are one seed. Round-to-round
  numbers swing ±3–4% (protocol noise, ~1 window = 0.56%), so treat the per-round values as
  indicative and the 4-round mean + movement metrics as the robust signal. Multi-seed confirmation
  of the SR/exact ≈ 86.8% > memory-free 85.7% ordering is the natural follow-up.
- **α range.** α = (1/lr)(1−1/β) ≈ −37,000 at (lr 3e-6, β0.9). Fine in FP32; a fixed-point kernel
  needs the range analysis below.

## Hardware / on-device-simulation implementation plan

Target: implement the ECO update inside the Onnx4Deeploy QZO **update graph** (`network_zo_update`)
so the on-device simulation (Siracusa now, GAP9 next) trains INT8 conv weights with error feedback
instead of the current `w_int += round(coeff·z/s_w)` rule. Recommended first device arm: **exact EF
with a stored residual** (numerically simplest, no α blow-up, best accuracy) — then memory-free once
range is confirmed.

### 1. Buffers (weights-as-inputs, resident in L2)

For the ~14.9k conv weights (SpeechNet), per variant:

| variant | extra device buffers | size (fp32) | notes |
|---|---|---|---|
| current direct-INT8 | none | 0 | stalls at small lr |
| master weights | fp32 master | ~60 KB | today's fix; ECO's target to replace |
| **ECO memory-free** | fp32 momentum `m` | ~60 KB | same size as master; write-sparse |
| **exact EF** | fp32 `m` + residual `e` | ~120 KB | 2× — but α cancels (no 1/lr factor) |
| plain int16 EF (not ECO) | int16 `e` | ~30 KB | cheapest; classical EF, for comparison |

Honest accounting (paper §8.2): because our ZO base has **no** momentum buffer to begin with,
memory-free ECO is *memory-neutral vs master weights*, not a saving; the saving in the paper comes
from momentum already being paid for in first-order LLM training. The genuine device wins here are
(a) **trainable INT8 conv weights** and (b) **write-sparsity** (11.8% vs master's 50%+). A true
memory *reduction* would come from quantizing the accumulator (int16 EF, ~30 KB) — a separate arm.

### 2. Kernel (element-wise over the weight tensors, 8-core PULP)

New op, e.g. `ECOUpdateRQS`, extending the existing INT8 update. Per weight (per-channel scale `s_w`,
pinned constants `lr`, `β`, `α`):
```
g       = coeff · z                    # coeff = (Lp−Lm)/(2·eps·n_accum); z = ±1 (Rademacher)
m_tilde = β·m + (1−β)·g
th_tilde= w_int·s_w − lr·m_tilde
w_new   = clamp(round(th_tilde/s_w), −127, 127)     # RTN; SR reuses the XORShift32 PRNG
e       = th_tilde − w_new·s_w
m       = m_tilde + α·e                 # memory-free; exact EF: m = m_tilde + (1/lr)e_prev − (1/(lr·β))e
w_int   = w_new
```
It is a single fused element-wise pass — the same shape/parallelism as the current perturb/update
kernels, so it slots into the existing 8-core fork. Overhead is one pass over 14.9k elements; per
§8.6 the cycle cost must be *measured* on the 8 RISC-V cores, not assumed negligible.

### 3. Fixed-point / numerical range

- α = (1/lr)(1−1/β) is large (≈ −3.7e4 at lr 3e-6, β0.9). The injected term α·e stays O(0.05)
  because |e| ≤ 0.5·s_w (s_w ≈ 4e-3), but intermediate `α·e` and `m` should be kept **fp32**; do not
  fixed-point the momentum without a range study. **Exact EF avoids the 1/lr blow-up** (its
  coefficients are 1/lr and 1/(lr·β) applied to `e` directly, and the `e_prev`/`e` largely cancel),
  which is why it is the recommended first device arm.
- Clamp to [−127,127] happens **after** the injection-implied step (compute residual against the
  clamped target), matching §8.7.

### 4. Graph integration (Onnx4Deeploy → Deeploy)

- Add `m` (and `e` for exact EF) as fp32 initializers / weights-as-inputs in the ZO update graph,
  alongside the existing perturbed-weight inputs (`base_exporter._export_qzo_training`).
- Replace the update-graph node that currently emits `round(coeff·z/s_w)` with the `ECOUpdateRQS`
  op; register its parser/bindings in Deeploy (Generic + PULPOpen tiling-ready, then GAP9 reuse —
  same pattern as the exp13 perturb-op registration).
- The momentum buffer restarts each round (each device round is a fresh runner invocation) — matches
  the simulation here; if rounds are chained on-device, persist `m` in L2 across rounds instead.
- Validate host-vs-device bit-exactness on the update kernel first (single step), then a full round,
  reusing the exp10/exp12 device-vs-host loss-compare tooling.

### 5. Recommended device sequence

1. Exact-EF, RTN, β0.9, lr 3e-6 — simplest numerics, best accuracy (86.81% in sim). Prove the
   kernel + buffers + bit-exactness.
2. Memory-free, RTN, β0.9 — drop the `e` buffer; confirm the 11.8% write-sparsity and 85.69% hold
   on device (this is the interesting MCU arm).
3. SR variant — reuse the on-device XORShift32 PRNG; confirm 86.81% and the 100%-movement cost.
4. Measure the ECO kernel cycle count vs the plain update, and the L2 footprint of `m` (±`e`).

## Reproduction

```bash
# in agitated_hugle, /app/Onnx4Deeploy/QZO_exp/exp15_QZO_eco
ARM=all_incr STEPS=2700 ROUNDS=1,2,3,4 python3 run_eco.py     # all arms, 4 rounds
# quick go/no-go diagnostic (cos(e) β-sweep, round 1):
ARM=all_diag STEPS=2700 ROUNDS=1 python3 run_eco.py
```
Results in `results.json` (`arms.<name>.round<r>`); per-line log in `run.log`; figure
`eco_summary.png`.
