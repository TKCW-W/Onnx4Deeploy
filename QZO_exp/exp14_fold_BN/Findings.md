# exp14 — Findings: BN folded into the quantized conv, PyTorch/Brevitas ZO simulation (2026-09-07 03:10 CEST)

Design under test (supervisor's): fold each block's BN into its int8 conv (W' = W·γ/σ, b' = (b−μ)·γ/σ + β, BN → Identity);
trainable = int8 conv W'/b' (+ fc); no fp32 BN parameters. Simulator: the exp_masterweight_ft Brevitas harness
(`fold_bn_sim.py` = harness + fold + recalibration + an LSB-domain ZO mode). Same recipe as all QZO sims: 2700 steps,
n_accum 4, seed 42, same z per regime, balanced accuracy on batch 2. Files: `results_*.json`, `sweep_*.log`, `run.log`.

## 1. Calibration must be re-done after folding — and it must not clip
| activation calibration (folded) | zero-shot |
|---|---|
| Brevitas observer abs-max (8 windows) | 78.33% |
| pooled@99.0 / 99.5 / 99.9 / 99.99 | 56.7 / 67.2 / 77.8 / 80.6% |
| **pooled@100 (exact max, 54 windows)** | **85.56%** = unfolded pooled@99.99 (85.56% true-int8) |
Folding moves the per-channel BN scale inside the conv, so the per-tensor int8 activation after the conv must span all
channels' post-BN ranges; clipping the tails is destructive; the exact max recovers the zero-shot fully.

## 2. Absolute eps (device-style `round(eps/s_w)` LSB) breaks after folding
Block 0's folded weights are tiny (|w|max 0.0114, s_w 1.7e-5–9e-5), so eps = 0.01 is **112–591 LSB** (> the 254-LSB grid);
every ±eps forward saturates block 0 → the ZO estimate is noise (`Plan.md` table). Measured (pooled@100):
direct 3e-6/1e-5/3e-5/1e-4 → 75.6 / 83.3 / 57.8 / 27.2%; master 3e-6/1e-5/3e-5 → 80.0 / 84.4 / 86.1%.

## 3. Scale-invariant LSB-domain ZO (perturb K of each weight's own steps, update in own steps)
| mode | K | lr_lsb | bal. acc | convW moved |
|---|---|---|---|---|
| direct_lsb | 1 | 1 / 3 | 85.56 / 85.56% | 0% (stall) |
| direct_lsb | 1 | 10 / 30 / 100 | 82.2 / 32.2 / 11.1% | 95–100% (random walk) |
| direct_lsb | 2 | 3 / 10 | 85.56 / 85.0% | 0% / 100% |
| master_lsb | 1 | 1 / **3** / 10 | 86.1 / **87.2** / 86.7% | 75 / 91 / 97% |
| master_lsb | 2 | 0.3 / 1 / 3 | 85.0 / 86.7 / 86.7% | 30 / 73 / 90% |

## 3b. fc-float folded variant (fc as fp32 `nn.Linear`, smooth float ZO path; conv LSB-domain; pooled@100, zero-shot 85.56%)
| regime | bal. acc |
|---|---|
| absolute eps: master@1e-5 / master@3e-5 / direct@1e-5 | 85.6 / 77.2 / 85.0% |
| LSB K=1: master_lsb@1 / direct_lsb@3 | **86.7%** / 85.6% (stall) |
A float fc does not restore the fine-tuning gain: the smooth trainable capacity the unfolded design has is the BN γ/β
(one gain+bias per channel in every block), not the fc.

## 4. Conclusion (relative comparison inside one simulator; Brevitas fake-quant is ~2 pt optimistic in absolute terms)
- **Folded BN + direct-int8 never beats zero-shot** at any eps parametrization or lr: the int8-only update either stalls
  (sub-LSB) or moves ~all weights as a whole-channel ±1-LSB random walk that degrades accuracy. There is no window between.
- **Folded BN + master weights** gains at most **+1.7 pt (87.2%)**, vs the **unfolded design's +4 (direct@1e-5, 88.3%) / +5
  (master@3e-6, 89.4%)** in the same harness. The unfolded design's gain comes mostly from the fp32 BN γ/β — a smooth,
  low-dimensional, well-conditioned path that ZO fine-tunes effectively and that folding removes.
- Comparable accuracy to the unfolded design was NOT reached with folded BN in this sweep (13 regimes over 3 eps
  parametrizations, 2 fc treatments, direct/master; best folded 87.2% vs unfolded 89.4%). If the supervisor's folded
  setup reaches higher, the difference must be in something not covered here (different
  perturbation/lr parametrization; larger n_accum; a different calibration/init) and should be pinned by reproducing his
  exact configuration in this harness.
