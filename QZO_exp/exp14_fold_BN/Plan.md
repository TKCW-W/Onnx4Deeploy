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

## Why absolute eps fails after folding (measured, pooled@100)
| param | folded s_w range | round(eps/s_w) LSB (eps=0.01) | \|w\|max | unfolded round(eps/s) |
|---|---|---|---|---|
| blocks.0.conv.weight | 1.7e-5 … 9.0e-5 | **112 … 591** (> the 254-LSB grid) | 0.0114 | 2 … 7 |
| blocks.1.conv.weight | 4.9e-4 … 1.3e-3 | 7 … 20 | 0.17 | 6 … 9 |
| blocks.2/3/4.conv.weight | 1.1e-3 … 6.4e-3 | 2 … 9 | 0.45–0.81 | 4 … 9 |
| conv biases / fc.bias | 7e-5 … 3.5e-3 | 3 … 152 | | 0 … 125 |
Folding scales block-0's weights by γ/σ = 0.005–0.06, so the device-style perturbation `round(eps/s_w)·z` with an absolute
eps saturates block 0 in every ±eps forward → ZO estimate is noise → training degrades (absmax: direct@1e-5 78.3→73.9;
pooled@99.99: direct@3e-6 80.6→65.6). Remedy under test: scale-invariant LSB-domain ZO (`direct_lsb`/`master_lsb`):
perturb each weight by K of its own steps, g=(L+−L−)/(2·K·n_accum), update round(−lr_lsb·g·z) LSB. Sweep at pooled@100:
K=1, lr_lsb ∈ {1,3,10,30,100} direct, {1,3,10} master → `results_pooled100_lsb1.json`.
