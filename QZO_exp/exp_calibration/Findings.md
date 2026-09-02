# exp_calibration — Findings

Date: **2026-09-02** · Branch `feat/QZO` · Ran in `agitated_hugle` (Brevitas 0.13.0) ·
Raw numbers: `results.json` · Log: `run.log` · Scripts: `calib_pooled.py`, `run_study.py`

## Headline

1. **Calibration quality matters enormously for the quantized forward** — the spread across
   procedures is 11.1% → 85.6% zero-shot balanced accuracy — and the corrected procedure
   (pooled percentile over all 1800 pretraining windows, percentile tuned) is the best of
   everything tested, at **99.99**.
2. **Calibration is NOT the cause of the LSB stall.** Under the best corrected calibration and
   under the old one, with shared probe seeds: the ZO signal is statistically identical
   (mean |g| 14.48 vs 14.55), **0.000%** of per-channel updates clear the 0.5 LSB rounding
   threshold at lr 3e-6, and in a 300-step direct-int8 fine-tune the conv int8 weights moved
   **exactly zero** times under *both* calibrations (`pct_steps_zero_conv_movement = 100.0`,
   `cum_pct_convw_int8_changed = 0.0`), while the master-weight control moved ~50.6–50.9% of
   conv int8 weights under both. The stall is invariant to activation calibration.

The hypothesis was re-opened without presuming the earlier refutations, tested with a properly
implemented calibration, and is now refuted at the mechanism level *and* the outcome level.

## 1 · What was wrong with our calibration, precisely (code-level, traceable)

Production call — `onnx4deeploy/core/base_exporter.py:706`:

```python
with torch.no_grad(), calibration_mode(model):
    model(calib)          # calib = 8 windows, session 3 batch 1
```

Three verified defects (Brevitas 0.13.0 sources, cited from the container install):

- **Per-batch percentile, not pooled.** `brevitas/core/scaling/standalone.py:456-475`
  (`ParameterFromRuntimeStatsScaling.training_forward`): each call computes
  `AbsPercentile` **of that batch only** and folds it into a running buffer
  (`inplace_momentum_update`) — a running *mean of per-batch percentiles* inside
  `calibration_mode` (momentum is set to `None`, `brevitas/graph/calibrate.py:479-480`), an
  EMA (momentum 0.1) outside it. Never a quantile of the pooled data.
- **The 99.999 percentile degenerates to the max per batch.** `AbsPercentile.forward`
  (`brevitas/core/stats/stats_op.py`) takes the k-th largest of the current batch; a batch of
  8 windows has ≤ 630k values at the largest site and ~46k at fc, so the 99.999th percentile
  is within ~6 values of the max (literally the max at the small sites). We were effectively
  doing per-batch abs-max calibration.
- **Wrong distribution.** The 8 (later 54) windows came from *session 3* — the target/held-out
  session — not the sessions 1+2 pretraining distribution the checkpoint was trained on.

The shipped reference offers no guidance here: `Onnx4Deeploy_ZO/onnx4deeploy/core/
base_exporter.py:539` (`export_zo_training`) runs `model.eval()` +
`input_tensor = torch.randn(*input_shape)` and never enters calibration; the collector is
gated on training mode, so every act scale stays at the uninitialised default threshold 1.0
→ scale 1/128 = 0.0078125 (`RescalingIntQuant`, `brevitas/core/quant/int.py:135`).

## 2 · The corrected procedure (implemented in `calib_pooled.py`)

- **Data**: all 1800 pretraining windows — S01/vocalized `sess_{1,2}_batch_{1..5}.h5`,
  180 each (asserted at load, `run.log:3-13`).
- **Streaming**: batch 64 → **29 collector steps** (28 full + one of 8), as reasoned.
- **Observation conditions**: identical to production — the hooks sit on
  `proxy.fused_activation_quant_proxy.tensor_quant` *inside* `calibration_mode(model)`, so the
  pooled collector sees exactly the tensors Brevitas' own collector sees (float network,
  quantizers in observer-only mode; `calib_pooled.py:156-180`).
- **Pooling**: |x| pooled per site across all batches (float16 store, 40M cap/site via uniform
  per-batch subsampling — only 1 of 12 sites needed the cap, at 27.8% kept; exact global max
  and exact N always tracked). **One** quantile per site at the end.
- **Freeze**: `ConstThreshold` surgery on each site's `scaling_impl`, verified bit-exact
  (`freeze_act_thresholds`, `calib_pooled.py:197-214`; the assert checks
  `proxy.scale() == t/128`).

### Was "99.999 basically the max"? (the framing question, answered by data)

Pooled, no — but on this data it is still deep inside an *artifact tail*. Block-0 input
(raw EMG), pooled over 17.64M values:

| statistic | 99.9 | 99.99 | 99.999 | max |
|---|---|---|---|---|
| threshold | 1,309 | 2,854 | 16,078 | 59,656 |
| rank from top | ~17,640 | ~1,764 | ~176 | 1 |

The top ~0.001% of the pretraining samples are ~45× above the 99.9th percentile — session 1+2
recordings contain extreme spikes that session 3 lacks (batch-2 max at the same site: 5,047).
So per-batch, 99.999 ≈ max (the old defect); pooled, 99.999 is a real tail statistic — but the
tail itself is pathological here, which is exactly why the percentile needed tuning.

## 3 · Sweep — calibration quality (zero-shot, whole session-3 batch 2, 180 windows)

| config | bal. acc | logit cos vs float | mean&#124;ΔCE&#124; | note |
|---|---|---|---|---|
| **pooled@99.99** | **85.56%** | 0.9919 | 0.140 | **best** |
| old8 (production, 8 sess-3 windows) | 84.44% | 0.9958 | 0.103 | |
| old54 (production, 54 sess-3 windows) | 82.78% | 0.9976 | 0.076 | |
| pooled@99.9 | 82.22% | 0.9968 | 0.073 | |
| float reference | 81.67% | 1 | 0 | |
| pooled@99.999 | 55.56% | 0.7833 | 1.115 | outlier-inflated |
| pooled@100 (abs-max) | 31.11% | 0.5965 | 2.504 | destroyed by the 45× tail |
| shipped (uncalibrated 0.0078125) | 11.11% | 0.2260 | 2.774 | = chance (1/9) |

Honest reading:

- The **decisive** contrasts are 99.99/99.9/old vs 99.999/100/shipped — orders of noise apart.
  The top-4 ordering is **within noise**: 1 window = 0.56% on this eval set, so
  85.56 vs 84.44 is 2 windows. A multi-fold/multi-seed sweep would be needed to rank them
  firmly.
- The two quality metrics disagree at the top: old54 is *closest to the float model*
  (cos 0.9976) but pooled@99.99 *classifies best*, 3.9 points above float itself.
  Fidelity-to-float and accuracy are different objectives; the differences are small-sample,
  so we note the disagreement rather than explain it away.
- Percentile verdict on the user's axis {99.9, 99.99, 99.999}: **99.99**. 99.999 is already
  inside the artifact tail (55.56%); 99.9 clips slightly too aggressively at the deeper sites
  but stays healthy.
- Two cross-checks: old8's 84.44% exactly reproduces the quantized zero-shot measured
  independently in `exp_masterweight_ft` (same procedure, different harness). And old54 scores
  *below* old8 — feeding the single-`calibration_mode`-call collector more windows does not
  improve it, consistent with the per-batch-max mechanism in §1.

## 4 · Stall linkage — the actual question

Protocol: 100 ZO probe steps at θ₀ (ε=0.01, the 54 stratified FT windows, shared z across
configs), then 300-step fine-tunes (lr 3e-6 — the best master lr from
`exp_masterweight_ft/results.json`), direct-int8 vs master-weight, under best-pooled vs old54.

**Probe signal — calibration-invariant** (`results.json:probe`):

| | pooled@99.99 | old54 |
|---|---|---|
| mean&#124;g&#124; | 14.48 | 14.55 |
| median&#124;g&#124; | 12.15 | 11.97 |
| max&#124;g&#124; | 52.5 | 53.9 |

**Implied update in LSB units** (per channel, `|lr·g·mul_factor|/s_w[c]`, 104 conv channels):

| lr | mean upd (LSB) | max upd (LSB) | frac (step,ch) ≥ 0.5 LSB | pooled vs old |
|---|---|---|---|---|
| 3e-6 | 0.029 | 0.16 | **0.0000 / 0.0000** | identical |
| 1e-5 | 0.097 | 0.53–0.55 | 0.0004 / 0.0006 | identical |

**300-step fine-tune** (`results.json:regimes`):

| run | conv int8 weights ever changed | steps with zero conv movement | final b2 bal. acc |
|---|---|---|---|
| pooled@99.99 · direct | **0.00%** | **100%** | 85.56% |
| old54 · direct | **0.00%** | **100%** | 83.89% |
| pooled@99.99 · master | 50.58% | 0% | 85.00% |
| old54 · master | 50.92% | 0% | 85.56% |

The direct runs' apparent accuracy motion comes **entirely from the fp32 parameters** (BN γ/β,
fc) that train alongside — the conv int8 weights are bit-frozen for all 300 steps under both
calibrations. This is the "falls back to partial training" regime, now shown to be unaffected
by calibration quality.

**Why, mechanically.** The stall criterion lives in the *weight* grid:
`|lr·g/s_w[c]| ≥ 0.5`. Activation calibration sets s_x only; s_w is data-free per-channel
abs-max (`max|W[c]|/127`), untouched by any calibration. The only path by which s_x could
matter is through g — and empirically the loss differences L₊−L₋ are the same size under both
calibrations (both forwards sit at cos ≈ 0.99 to the float model; the probe distributions
overlap completely). To clear 0.5 LSB at |g|≈14.5 with s_w ∈ [0.0018, 0.0036] would need
lr ≈ 6e-5–1.2e-4, i.e. 20–40× the lr that fine-tuning tolerates (from
`exp_masterweight_ft`: 3e-6 best; larger already degrades) — a calculation, not a tested
regime.

## 4b · Round-1 fine-tune at full length (added 2026-09-02, `run_round1.py`)

Same simulation, full round-1 recipe (**2700 steps** = 200 epochs × 54 windows / n_accum 4,
lr 3e-6, ε 0.01 — the exp18/exp5 recipe), batch-2 eval:

| run | final b2 bal. acc | conv int8 changed | zero-move steps |
|---|---|---|---|
| **pooled@99.99 · direct** | **87.78%** | **0.00%** | 100% |
| pooled@99.99 · master | 87.22% | 76.32% | 0.2% |
| old54 · direct | 85.56% | 0.00% | 100% |
| old54 · master | 88.33% | 76.51% | 0.3% |

Context: float-ZO round-1 PyTorch sim = 87.36% (exp18), float zero-shot = 80.56%,
quantized zero-shot pooled@99.99 = 85.56%.

Reading (honest): at round-1 scale on this task, **direct-int8 matches master-weight and
float ZO in accuracy** (all ≈ 87–88%, spread ≤ 1.1 pt ≈ 2 eval windows) — *while its conv
int8 weights never move once in 2700 steps*. The adaptation this task needs is delivered
by the fp32 BN γ/β + fc parameters alone. So "direct int8 ≈ float ZO" is reproduced here,
but the movement counter shows it is BN/fc partial training in disguise, not int8 weights
learning; the accuracy cost of the stall is not measurable in one round of this bench.
Master's generality (conv weights genuinely train, 76% of them changed) buys no measurable
round-1 accuracy on this task — the case for master weights rests on generality across
tasks/rounds, not on this benchmark's round-1 number.

## 5 · Verdict and what stands

- **Calibration is refuted as the cause of the LSB stall** — now with the properly implemented
  calibration, tested at both the mechanism (probe/LSB) and outcome (300-step FT) level, with
  shared randomness. Master weights remain the solution for training conv weights.
- **The corrected calibration procedure is worth adopting regardless**: pooled percentile
  @ 99.99 over the full pretraining set is the best-scoring configuration and is a defensible,
  batching-independent data statistic — unlike the current 8-window per-batch-max EMA, which
  scored 84.44% by luck of construction (its per-batch max ≈ pooled 99.99–99.999 range on
  session-3 data, see thresholds table in `results.json:pools/old_scales`).
- Follow-up (not yet done): port pooled@99.99 pretraining-data calibration into the production
  export path at `base_exporter.py:706` (comment out the old call, keep alongside), and rerun
  the FT accuracy study with it; a multi-fold/multi-seed sweep to rank the top-4 configs
  beyond the 180-window noise floor.

## Caveats (stated, not hidden)

- Host simulation with the device update rule; single subject/fold (S01, fold 3), single seed;
  180-window eval (1 window = 0.56%).
- The quantized-beats-float result (85.56 vs 81.67) is 7 windows on a small set — noted,
  not interpreted.
- `pooled@99.99|direct` touched 87.78% mid-run (step 150, `run.log:63`) — trajectory noise
  from the fp32-parameter training; final values are the honest comparison point.

## Files

```
Plan.md                 the pre-registered design (2026-09-02)
calib_pooled.py         pooled-percentile calibration (reusable; Brevitas internals cited inline)
run_study.py            staged, resumable study driver (data/pools/sweep/probe/regimes/plots)
results.json            every number in this document
run.log                 full run transcript
scales_per_site.png     thresholds per site per config (log scale)
quality_metrics.png     sweep quality metrics
lsb_clearance.png       LSB-clearance distributions, both calibrations
regime_trajectories.png 300-step FT trajectories, 4 runs
data_cache.npz / pools_cache.npz   stage caches (not committed; regenerate via run_study.py)
```
