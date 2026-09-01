# exp_calibration — Findings draft (2026-09-02)

All numbers below are from `results.json`, produced by `run_study.py` on 2026-09-02
(container `agitated_hugle`, Brevitas 0.13.0). Draft for review; every code claim carries
its file:line. Study contract: `Plan.md`.

## 1. What was actually run

- **Proper pooled calibration** (`calib_pooled.py`): S01/vocalized **sessions 1+2, all 10
  h5 batches = 1800 windows** (asserted at load, `run_study.py stage_data`; each
  `sess_{1,2}_batch_{1..5}.h5` yields exactly 180 windows), streamed as **29 collector
  batches of 64** (28x64+8). |x| pooled per activation-quantizer site (12 sites), one
  quantile per site at the end. Only `blocks.0.conv.output_quant` exceeded the 40M cap
  (141.3M values -> 27.8% uniform per-batch subsample kept; exact max always tracked
  separately). Observation ran inside Brevitas' own `calibration_mode`, so the observed
  tensors are exactly what the stock collector sees (float network, quant disabled,
  observer-only proxies — brevitas/graph/calibrate.py:463-485 and :151-179).
- **Configs swept** (scales frozen bit-exactly via `ConstThreshold` surgery, verified by
  assertion in `calib_pooled.py:freeze_act_thresholds`): pooled@{99.9, 99.99, 99.999,
  100=abs-max}, `old8` / `old54` (the production single-call `calibration_mode` procedure of
  `onnx4deeploy/core/base_exporter.py:706` with the first 8 / all 54 FT windows), `shipped`
  (uncalibrated default scale 1/128 everywhere), and the float model.
- **Stall linkage**: 100 ZO probe steps at theta0 (eps=0.01, n_accum=4, seed 42, shared z
  across configs) + 4 fine-tune runs (direct-int8 and master-weight x best-pooled and old54
  calibration), **300 steps** each at lr 3e-6 (the best master lr from
  exp_masterweight_ft), eval every 50 steps. 300 steps (not the full 2700-step recipe) was
  sufficient because the question is *whether the int8 weights move at all*.

## 2. Calibration quality (zero-shot, whole session-3 batch-2, 180 windows)

| config | bal. acc | logit cos vs float | mean\|dCE\| | comment |
|---|---|---|---|---|
| **pooled@99.99** | **85.56%** | 0.9919 | 0.140 | best — recommended |
| old8 (production, 8 windows) | 84.44% | 0.9958 | 0.103 | = masterweight_ft zero-shot, exact reproduction |
| old54 (production, 54 windows) | 82.78% | 0.9976 | 0.076 | more data did NOT help the single-call collector |
| pooled@99.9 | 82.22% | 0.9968 | 0.073 | best float-fidelity, not best accuracy |
| pooled@99.999 | 55.56% | 0.7833 | 1.114 | collapsed — outlier-inflated scales |
| pooled@100 (abs-max) | 31.11% | 0.5965 | 2.504 | collapsed |
| shipped (uncalibrated 1/128) | 11.11% | 0.2260 | 2.774 | chance level (9 classes) — confirms the shipped export is unusable |
| float model | 81.67% | — | — | reference |

Key observations:

- **The pretraining EMG data contains extreme outliers**: pooled max at the raw-input site
  is 59,656 (sessions 1+2) vs 5,047 on session 3 — a 12x tail. That is why 99.999/100
  percentiles produce catastrophic scales here: this is not the textbook "high percentile
  ~ abs-max is fine" regime. The percentile axis matters, and the sweet spot is **99.99**.
- To the framing question "*is 99.999 basically the max?*": per-batch yes (rank ~ max in a
  64-window batch); pooled over 1800 windows it is a genuine tail statistic (rank ~176 of
  17.6M at the largest site) — but on this data even the genuine 99.999 tail is
  outlier-dominated, so it loses anyway.
- The production procedure was neither EMA nor average in practice: `base_exporter.py:706`
  makes **one** `model(calib)` call, so the collector state is just the 99.999-percentile
  of that single batch (~ its max; `ParameterFromRuntimeStatsScaling.training_forward`,
  brevitas/core/scaling/standalone.py:456-475, folds per-CALL `AbsPercentile` stats —
  a mean of per-batch percentiles at best, never a pooled quantile).
- Float-fidelity (cosine, |dCE|) and accuracy **rank configs differently** (old54 has the
  best cosine and the 3rd-worst accuracy of the sane configs). With 180 eval windows,
  1 window = 0.56% overall; accuracy gaps of 1-3% are a few windows and single-fold.
  The 30-74% gaps (collapsed configs) are unambiguous.
- Scale convention note: Brevitas act scale = threshold/128
  (`RescalingIntQuant.forward`, brevitas/core/quant/int.py:156-160 — the scaling impl
  receives `int_threshold=128` and divides internally; the plan's `q_p/127` was off by
  this convention detail).

## 3. Stall linkage — does proper calibration change the stall? **No.**

ZO probe signal at theta0 (100 shared-z steps, eps=0.01):

| | pooled@99.99 | old54 |
|---|---|---|
| mean \|g\| | 14.479 | 14.546 |
| implied update, mean [LSB] (lr 3e-6) | 0.0290 | 0.0291 |
| implied update, max [LSB] (lr 3e-6) | 0.160 | 0.164 |
| (step,channel) pairs >= 0.5 LSB (lr 3e-6) | **0.000%** | **0.000%** |
| (step,channel) pairs >= 0.5 LSB (lr 1e-5) | 0.038% | 0.058% |

The ZO gradient magnitude is insensitive to which (sane) calibration is installed —
0.5% difference in mean |g|. No step clears the rounding threshold at the working lr under
either calibration.

Behavioral confirmation (300-step fine-tune, lr 3e-6, shared z/windows):

| run | final bal. acc | steps with zero conv-weight movement | conv int8 weights changed |
|---|---|---|---|
| pooled@99.99 . direct-int8 | 85.56% (= its zero-shot, nothing moved) | **100.0%** | **0.000%** |
| pooled@99.99 . master | 85.00% | 0.0% | 50.58% |
| old54 . direct-int8 | 83.89% | **100.0%** | **0.000%** |
| old54 . master | 85.56% | 0.0% | 50.92% |

Under the *best* calibration found, the direct-int8 update still moves **0 of 14,880 conv
weights in 300 steps**, exactly as under the old calibration; the master-weight regime
moves ~50% of them under either. (Direct runs still drift their int32 biases/BN, hence the
small accuracy wiggle.)

**Conclusion supported by the data:** activation-scale calibration quality is a real,
separate issue (it moves zero-shot accuracy by up to 74 points across configs and the
recommended procedure improves it by ~1.1 points over production), but it is **not the
cause of the LSB stall and cannot fix it** — the stall lives in the weight grid
(s_w = max|W[c]|/127, data-free, untouched by any activation calibration), and the probe
statistics that could in principle have changed (|g|) do not change. This closes the
re-examination requested after the earlier refutations, this time with a properly
implemented collector.

## 4. The proper calibration procedure (implementation-level deliverable)

`calib_pooled.py` — pooled-percentile PTQ calibration, reusable:
pretraining data (1800 windows, sessions 1+2), batch 64 / 29 steps, pooled |x| per site
(cap 40M/site, exact max tracked), one quantile at **99.99**, frozen as constant thresholds
(`ConstThreshold`, matching the RescalingIntQuant call contract, verified bit-exact).
Recommended port into the production export path (`base_exporter.py` calibration site,
comment-out-don't-delete) as a follow-up.

## 5. Artefacts

`results.json` (all numbers incl. per-site scales, sat/zero rates, g samples),
`run.log`, `scales_per_site.png`, `quality_metrics.png`, `lsb_clearance.png`,
`regime_trajectories.png`, caches (`data_cache.npz`, `pools_cache.npz`).
Saturation/zero-bin rates per config are in `results.json -> sweep -> <config> ->
site_health` (e.g. pooled@99.99 vs old54: block-0 zero-bin 0.42 vs 0.475 — both scales
are coarse for the raw input because the input itself is heavy-tailed).
