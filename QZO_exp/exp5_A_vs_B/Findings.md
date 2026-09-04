# exp5_A_vs_B — Why the Brevitas simulation and the integer datapath train to different places

Date: **2026-09-04** · `matched_z_check.py`, `b_perturbation.json`, `matched_z_losses.json`
Branch `feat/QZO`

## The question

On round 1 of QZO fine-tuning (S01/vocalized/fold 3, lr 1e-5, 2700 steps, pooled@99.99, ε 0.01,
seed 42) two implementations of *the same intended algorithm* disagree on the outcome:

| implementation | zero-shot b2 | after round 1 |
|---|---|---|
| **A** — PyTorch + Brevitas *fake* quantization (fp32 snapped to grid) | 85.00% | **88.89%** (+3.89) |
| **B-host** — `run_onnx_graph` on the exported **true integer** graph | 85.00% | **83.89%** (−1.11) |
| **B-device** — the same integer graph compiled to C, on GVSoC | (= 85.00) | 83.33% predicted / **82.95% measured** |

B-device reproduces B-host to within **one eval window**, and their per-step losses are bit-exact
for 600+ steps (divergence begins at 1 ulp = 1e-6 and amplifies exponentially — floating-point
chaos, not a computational difference; see exp9). **So the device is not the problem.** The
problem is that A and B train to opposite outcomes from an identical starting point.

## Method — removing the confound

A and B use different Rademacher streams (numpy `RandomState(SEED+u)` vs the device xorshift32
per node_id), so comparing trajectories is confounded: two *correct* implementations would still
diverge. We removed the confound by extracting **B's actual z** (running B's perturb nodes alone
and taking `sign(perturbed − original)`; deterministic given seed/node_id) and feeding that exact
z into A, then comparing from identical weights on identical mini-batches.

Two controls validate the comparison:

- **Control 2 (test validity)** — the applied per-channel perturbation magnitudes match:
  B `[9, 6.95, 7.94, 5.95]` vs A `[9, 7, 8, 6]` (blocks 1–4 within rounding jitter; block 0
  per-channel `[3, 6, 3, 2]` identical). z extraction and tensor-layout mapping are correct.
- **Control 1 (forward agreement)** — *unperturbed* loss on the same 4 mini-batches:

  | | mb0 | mb1 | mb2 | mb3 |
  |---|---|---|---|---|
  | A | 0.377449 | 0.536897 | 0.042880 | 0.522462 |
  | B | 0.342002 | 0.575994 | 0.037004 | 0.508196 |
  | \|diff\| | 3.5e-2 | 3.9e-2 | 5.9e-3 | 1.4e-2 |

## The result

Same weights, same z, same data:

| | mb0 | mb1 | mb2 | mb3 | **g = Σ(L₊−L₋)/(2ε·n_accum)** |
|---|---|---|---|---|---|
| A · L₊−L₋ | +1.123 | −0.797 | +0.080 | +1.140 | **+19.32** |
| B · L₊−L₋ | +0.795 | −1.368 | +0.028 | +0.400 | **−1.81** |

**The gradient estimates have opposite signs and differ by an order of magnitude.**

## Interpretation — no bug is required

ZO learns from *nothing but differences of losses at perturbed points*. Control 1 shows A and B
already disagree by 0.006–0.039 on the unperturbed loss; under the ±3–9 LSB perturbation that
disagreement grows to 0.3–0.7 per pair, which is the same size as the signal itself. Two
legitimate quantized implementations differing only in rounding details (truncating RequantShift
vs fake-quant rounding, round-half-up in the fixed-point perturb kernel, ±127 clamping) therefore
estimate **different gradients from the same random direction**. Over 2700 steps that fully
accounts for +3.89 vs −1.11 without postulating an error anywhere.

The uncomfortable corollary, stated plainly:

> **Matching accuracy (85.00 = 85.00) and logit cosine (0.9994) at the operating point is NOT
> sufficient fidelity to predict ZO training.** Cross-entropy is far more sensitive than argmax,
> and the ZO signal is a difference of two such quantities at perturbed points. A fake-quant
> simulator can therefore agree on inference and still disagree on training — even in the sign of
> the gradient.

Consequence for this project: the hyperparameters tuned in A — the lr sweep (1e-5), the
calibration percentile (pooled@99.99), the cross-fold stability study, the master-weight /
stochastic-rounding / ceiling comparisons — were all optimized on a loss landscape that is **not
the deployed one**. B is the deployment reality. The 88.89% was never available on device.

## Secondary defects found in the audit (real, but not the main cause)

1. **Antithetic symmetry is broken** for the int8 conv weights: `dp + dm ≠ 0` on ~1% of elements
   in blocks 1–4 and **25%** in block 0 (|dp| 3.66 vs |dm| 2.91). Mean bias is small
   (0.007–0.19 LSB on ~7 LSB steps). Dominant mechanism is ±127 **clamping** (which A shares),
   with a contribution from the kernel's round-half-up `(z·mul + 2^14) >> 15` (asymmetric for
   negative z under arithmetic shift). A centred estimator wants `+δ` and `−δ` exactly.
2. **`blocks.0.conv.bias_rqsadd` receives exactly zero perturbation** (|dW| = 0.0000) — the
   block-0 bias can never train. Its `s_b = s_in·s_w` is ~300× coarser than the other blocks'
   (block-0's input scale is 22.3 from the raw-EMG calibration), so `round(ε/s_b) = 0`.
   Consistent with the per-tensor movement audit (block-0 bias 0% moved).

## What this changes

The right question is no longer *"how do we make the device reach 88.89%?"* — that target was a
simulation artifact. It is:

> **Can QZO improve accuracy on the true integer datapath at all, and under what settings —
> tuned on B, not on A?**

Next steps, cheapest first, all on B-host (B-device is diagnostically redundant: it reproduces
B-host to one window at hours-per-iteration cost):

1. **Multi-seed B** at the current setting — is −1.11 systematic or one unlucky draw? (B has been
   run exactly once.) *[in progress]*
2. **lr sweep on B directly** — the threshold rule gave 1.02e-5 from B's own |g| distribution, but
   that rule was validated in A; sweep it on B.
3. **Fix the two defects** — symmetric rounding in the perturb kernel, and a workable bias grid
   for block 0 — removing known bias from the estimator.

## Reproduction

```bash
docker exec agitated_hugle bash -lc 'cd /app/Onnx4Deeploy/QZO_exp/exp5_A_vs_B && python3 matched_z_check.py'
```
Outputs `b_perturbation.json` (per-tensor perturbation audit: magnitudes, antithetic check,
zero-step counts) and `matched_z_losses.json` (the matched-z A/B loss pairs and g).
