# exp14 — BN folded into the quantized conv (supervisor's design): PyTorch/Brevitas ZO simulation + config sweep

Started 2026-09-07 02:15 CEST. Question: with BatchNorm FOLDED into the requantized conv (no fp32 BN params; trainable =
int8 conv W'/b' + fc), can QZO reach accuracy comparable to the unfolded design (unfolded: Brevitas 88.89% @1e-5,
true-int8 86.67%; float ZO ~87–88%)? BN folding changes the conv weights and the post-conv activation distributions,
so weight AND activation scales are re-derived (user's note).

## Method (`fold_bn_sim.py`, runs in agitated_hugle)
Reuses the exp_masterweight_ft harness verbatim (`mw.py` copy: model = pretrained Brevitas QuantSpeechNetDeploy, real-data
calibration, frozen scales, ZO loop with direct-int8 / master / stochastic modes, balanced accuracy on batch 2).
One inserted step: right after the pretrained model is built and BEFORE calibration, each block's fp32 BN (checkpoint
running stats, γ, β) is folded into its QuantConv2d: `W' = W·γ/σ`, `b' = (b−μ)·γ/σ + β`, BN → Identity. Per-channel
int8 weight scales are then derived from W'; activation scales re-calibrated on the folded model (`CALIB`):
`absmax` (Brevitas observers, CALIB_SAMPLES windows) or `pooled@P` (pooled |x| percentile P over the 54 training windows
at the 12 act sites, `exp_calibration/calib_pooled.py`, frozen via ConstThreshold) — applied BEFORE the trainable grids
are built (the int32 bias grid is s_in·s_w). Trainable params: 12 (5 conv W', 5 conv b', fc W, fc b), 0 fp32.
Recipe as before: 2700 steps, n_accum 4, eps 0.01, seed 42, same z draw across regimes.

## Calibration scan (folded, zero-shot, batch 2)
| CALIB | zero-shot | note |
|---|---|---|
| absmax (observer, 8 windows) | 78.33% | |
| pooled@99.0 / 99.5 / 99.9 | 56.7 / 67.2 / 77.8% | clipping the post-BN activations is destructive |
| pooled@99.99 | 80.56% | |
| **pooled@100** (exact max, 54 windows) | **85.56%** | = unfolded pooled@99.99 (85.56% true-int8) — fold fully recovered |
(unfolded, same harness, absmax: 84.44%)

## Sweeps
`REGIMES=direct@{3e-6,1e-5,3e-5,1e-4},master@{3e-6,1e-5,3e-5}` at CALIB ∈ {absmax, pooled@99.99, pooled@100} →
`results_<calib>.json`, `sweep_<calib>.log`, `run.log`. Compare with `exp_masterweight_ft/results.json` (unfolded).

## Commands
```
EXTRA=x CALIB=pooled@100 RESULTS=results_pooled100.json REGIMES=direct@3e-6,... python3 fold_bn_sim.py
EXTRA=x CALIB=pooled@99.9 RESULTS=zs.json REGIMES= python3 fold_bn_sim.py      # zero-shot only
```
Caveat (as in exp10): the Brevitas fake-quant sim is ~2 pt optimistic vs the true int8 datapath; used here for the
RELATIVE comparison folded vs unfolded under one simulator. fc is quantized (harness convention), unlike the device's fc-float.
