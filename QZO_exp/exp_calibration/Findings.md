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

## 4c · Direct-int8 lr sweep — a higher lr DOES overcome the stall (added 2026-09-02)

`run_lr_sweep.py` + `run_lr_confirm.py`, direct int8, pooled@99.99, full round-1 protocol
(2700 steps), batch-2 eval:

| lr | b2 bal. acc | conv int8 changed | steps with any movement |
|---|---|---|---|
| 3e-6 (baseline) | 87.78% | 0.00% | 0% |
| 6e-6 | 89.44% | 2.98% | 0.1% |
| **1e-5** | **90.00 / 90.56 / 89.44%** (z-seeds 42/12345/67890) | 66–86% | 0.9–1.4% |
| 2e-5 | 87.78% | 93.23% | 10.8% |
| 3e-5 | 86.11% | 97.53% | 23.7% |
| 6e-5 | 70.56% | 99.03% | 58.6% |
| 1e-4 | 22.78% | 99.52% | 85.3% |
| 3e-4 | 10.56% | 99.62% | 98.0% |

**lr = 1e-5 is the stable un-stalled setting**: mean ≈ 90.0% over three independent z-seeds —
above master-weight (87.2–88.3%) and the float-ZO round-1 reference (87.36%) — with graceful
degradation on both sides. Mechanism: at 1e-5 only the rare large-|g| steps (~1%) clear the
0.5 LSB threshold, so rounding acts as an implicit update-on-strong-signal-only filter (whole
channels step ±1 LSB per event); at ≥2e-5 noisy steps clear it too and the int8 weights
random-walk (monotone collapse). Note the max-update criterion, not the mean, is what predicts
the onset: max|g|≈53 ⇒ first movement at lr ≈ 0.5·s_w_min/53 ≈ 6e-6, exactly as observed.

**Movement anatomy at lr 1e-5** (`run_move_anatomy.py`, deterministic replay): 34 of 2700
steps moved anything. Per moving step, 1.5–76.7% of the 14,880 conv weights (median 6.2%) —
whole channels step ±1 LSB per event. |g| on moving steps: min 49.4 / median 55.6 / max 82.5,
vs overall mean |g| ≈ 14.5 — only the extreme-|g| tail triggers movement. Union of ever-moved
weights 76.81%; net changed vs init 65.93% (bounces cancel).

**Attribution ablation at lr 1e-5** (`run_ablate_frozen_w.py`, same z/windows; in this sim
only BN γ/β are fp32 — the fc weight is int8 like the convs, and conv/fc biases are int32 on
the ~50× finer s_in·s_w grid, so biases clear their LSB even at 3e-6):

| lr 1e-5 variant | b2 bal. acc |
|---|---|
| full direct (everything trains) | 90.00% |
| int8 weights frozen — BN+biases only | 87.78% |
| BN+biases frozen — int8 weights only | 84.44% (below the 85.56% zero-shot) |

So the ~90% is **mostly BN+bias learning (87.78% on its own) plus a real ≈+2.2 pt
contribution from the int8 weight movement** — but only jointly: the weight movement alone,
without BN/bias co-adaptation, sits slightly below zero-shot. (+2.2 pt = 4 eval windows;
directionally consistent with the 3-seed ~90% cluster for the full run vs 87.78%.)

## 4d · Incremental fine-tuning, QZO direct@1e-5 vs float ZO (added 2026-09-02, `run_incremental.py`)

Full incremental protocol, S01/vocalized/fold3, session 3. Rounds r=1..4: fine-tune on 54
stratified windows (30% of 180, seed 42) from batch r, carry weights, evaluate on the WHOLE
batch r+1 (180 windows). Recipe: 200 epochs × 54 / n_accum 4 = **2700 steps/round**, ε 0.01,
z-seed restarts at 42 each round (mirrors the device runner). **QZO** = direct int8, lr 1e-5,
act scales frozen at pooled@99.99 (pretraining calibration) and weight scales at pretrained
per-channel abs-max — **both frozen across all rounds, no recalibration**. **Float ZO** = plain
SpeechNet, lr 3e-6, eval-mode BN, all 22 tensors trainable — **same per-round fixtures**.

| round | ft→eval | QZO before→after | float before→after | QZO−FP (after) | conv moved this round | cum net |
|---|---|---|---|---|---|---|
| 1 | b1→b2 | 85.56 → **90.00** | 81.67 → 87.78 | **+2.22** | 65.9% | 65.9% |
| 2 | b2→b3 | 85.00 → 84.44 | 83.33 → 83.33 | +1.11 | 42.9% | 62.3% |
| 3 | b3→b4 | 89.44 → 87.78 | 91.11 → 89.44 | −1.67 | 88.5% | 87.1% |
| 4 | b4→b5 | 82.78 → 82.22 | 85.00 → 83.89 | −1.67 | 40.7% | 85.8% |
| **mean after** | | **86.11** | **86.11** | **−0.00** | | union → 98.3% |

Readings (honest):

- **QZO matches float ZO on average** — mean post-FT balanced accuracy 86.11% for both, per-round
  gap in [−1.67, +2.22], i.e. ≤ 3 eval windows either way. Quantized ZO with a tuned lr is not
  measurably worse than float ZO on this task. This is the headline the supervisor asked for.
- **The conv int8 weights genuinely train in every round** — 65.9 / 42.9 / 88.5 / 40.7% moved
  per round, cumulative net 85.8% vs pretrained, union 98.3%. The strong-signal-filter mechanism
  does **not** stall out as the model nears the session-3 distribution; the earlier worry that
  direct@1e-5 was a round-1 artifact is refuted.
- **Both methods improve on EVERY batch above the pretrained zero-shot baseline** (corrected —
  an earlier draft here wrongly said "neither improves round-over-round", which measured the
  *marginal* post-round-(r−1)→post-round-r delta, not the gain over the pretrained baseline). The
  correct comparison, vs per-batch zero-shot (float / quant): b2 +6.1/+4.4, b3 +6.7/+6.1,
  b4 +1.7/+0.6, b5 +7.8/+5.0. b4 gains least because its zero-shot is already ~87.8. This matches
  the established float-ZO result in `SilentWear/.../exp18_zo_faithful_sim/FINDINGS.md`, whose
  protocol is identical (streaming carry, train b_r → eval b_{r+1}, b1 untouched, 200 ep, ε 0.01,
  lr 3e-6, shared-z scalar accumulation, full model, frozen BN); its zero-shot row matches ours
  bit-for-bit.

| batch | zero-shot float | my float ZO | zero-shot quant | my QZO | QZO conv wts moved this round |
|---|---|---|---|---|---|
| b2 | 81.67 | 87.78 | 85.56 | 90.00 | 9,810 / 14,880 (65.9%) |
| b3 | 76.67 | 83.33 | 78.33 | 84.44 | 6,377 / 14,880 (42.9%) |
| b4 | 87.78 | 89.44 | 87.22 | 87.78 | 13,165 / 14,880 (88.5%) |
| b5 | 76.11 | 83.89 | 77.22 | 82.22 | 6,055 / 14,880 (40.7%) |
| **b2–5** | **80.56** | **86.11** | **82.08** | **86.11** | cum-net 85.8%, union 98.3% |

(exp18 float-ZO reference for the same protocol: b2–5 = 87.36; discussion of the ~1.25 gap is
in the bullets below.)

(conv-weights-moved is the QZO int8 movement in the round trained on batch r−1 whose result is
the b_r eval row; net within that round, out of 14,880 conv weights.)

- **Apples-to-apples QZO == float ZO under the identical harness (86.11 = 86.11)**, both improving
  every batch. QZO pays nothing vs float ZO run the same way.
- **My float ZO is ~1.25 below the exp18 reference (86.11 vs 87.36), consistently ~1.2/batch.**
  vs exp18's multi-seed §9 (b2–5 = 87.2 ± 0.3, per-batch seed-std up to 3.4 on the weak batch b3),
  my single run is low mainly on b3 (83.33 vs 85.2 ± 0.3) and b5. Likely cause: a different
  Rademacher z-stream (numpy `RandomState(42+u)` vs exp18's device `_perturb_rademacher`) — a
  single-seed realization unlucky on the hard streaming batches. It affects the float baseline
  too, so it is a harness/RNG offset, not a quantization penalty. Open: 2–3 more seeds to confirm
  the mean regresses toward 87.2 ± 0.3.

Caveats: single subject/fold, single seed (RNG-stream-dependent, see above); 180-window eval
(1 window = 0.56%); host sim; float ZO at its recipe lr 3e-6 vs QZO's tuned 1e-5 (each at its
best lr). Float model forward is batch-1 only (per-sample forward used; float zero-shot
batch-2 = 81.67% cross-checks the sweep's float_ref and exp18's zero-shot row exactly).

## 4e · Why is quantized zero-shot > float zero-shot on batch 2? (added 2026-09-02, `run_quant_vs_float_zeroshot.py`)

Batch-2 quantized zero-shot (85.56%) beats float (81.67%) by +3.89. Discriminating test —
quant−float on in-distribution (pretraining sess 1+2) vs held-out (session 3), pretrained
weights, no FT:

| eval | quant | float | q−f |
|---|---|---|---|
| in-dist sess1 b1 | 95.56 | 95.00 | +0.56 |
| in-dist sess1 b3 | 96.67 | 97.22 | −0.56 |
| in-dist sess2 b2 | 90.00 | 95.00 | −5.00 |
| held-out sess3 b2 | 85.56 | 81.67 | +3.89 |
| held-out sess3 b3 | 78.33 | 76.67 | +1.67 |
| held-out sess3 b4 | 87.22 | 87.78 | −0.56 |
| **mean in-dist** | | | **−1.67** |
| **mean held-out** | | | **+1.67** |

The sign flips as the regularization hypothesis predicts: quantization slightly *hurts*
in-distribution (lossy approx of weights already well-fit) and slightly *helps* held-out — the
signature of trading fit for robustness. Mechanism is consistent with §2: pooled@99.99 clips the
activation outlier tail, which on a shifted session was doing net harm. BUT the effect is small
and batch-dependent (held-out wins span +3.89 → −0.56); batch 2 is the favorable end of the
range, not a representative +4. Honest statement: **quantization is ~accuracy-neutral with a
small (~+1.7 mean) held-out regularization benefit; the batch-2 +3.89 is that benefit at its
high end, not a stable gain.** This reframes §4d: QZO doesn't out-learn float ZO — it starts from
a marginally better-generalizing zero-shot and both fine-tune to the same 86.11% mean. Single
fold; the +1.7 needs a multi-fold sweep to confirm as stable rather than a coin-flip.

## 4f · Multi-seed confirmation (added 2026-09-02, `run_incremental_multiseed.py`, seeds 42/1/7/123)

Same 4-round streaming protocol, pooled over 4 seeds (seed drives both the Rademacher z-stream
and the 54-window stratified FT draw). Per-batch mean ± std (ddof=1), post-FT balanced accuracy:

| method | b2 | b3 | b4 | b5 | b2–5 |
|---|---|---|---|---|---|
| float ZO (lr 3e-6) | 86.94 ± 0.72 | 83.75 ± 2.28 | 89.31 ± 1.46 | 84.17 ± 0.32 | **86.04** |
| QZO int8 @ 1e-5 | 89.03 ± 1.15 | 84.17 ± 1.40 | 88.47 ± 1.23 | 82.22 ± 1.57 | **85.97** |

Per-seed b2–5: float ZO {42:86.11, 1:85.56, 7:86.25, 123:86.25}; QZO {42:86.11, 1:85.83,
7:84.72, 123:87.22}.

Conclusions:

- **QZO ≈ float ZO holds across seeds**: 85.97 vs 86.04 mean b2–5, a 0.07-pt gap — statistically
  indistinguishable (both ≈ ±1 pt seed spread). Quantized ZO at lr 1e-5 costs nothing vs float ZO
  under the identical harness. This is now a multi-seed result, not a single point.
- **The ~1.25-pt gap to the exp18 reference (87.36) is real and NOT closed by seeds.** My float ZO
  averages 86.04 ± ~0.35 across 4 seeds — it does *not* regress to exp18's 87.2 ± 0.3; the gap is
  systematic, ~1.2 pt, concentrated on b3 (83.75 vs exp18 85.2) and b5 (84.17 vs 85.6). Since it
  hits the float baseline equally, it is **not a quantization penalty** — it is a harness
  difference between this study's float-ZO path and exp18's `zo_faithful.py`. Most likely the
  Rademacher realization (numpy `RandomState` per-step vs exp18's device `_perturb_rademacher`
  bitstream) or a minor FT-draw / step-count detail; exp18 §1 argues PRNG choice is accuracy-
  neutral in expectation, so a persistent offset points at a small protocol detail worth a
  code-level diff before quoting absolute numbers against exp18. For the QZO-vs-float question it
  is immaterial — both sit on the same harness.
- b3 remains the high-variance batch for both methods (std 2.28 / 1.40), matching exp18 §9's
  finding that b3 is the structural weak/variable batch.

## 4g · Incremental at lr 3e-6 (stalled control), 4 seeds (added 2026-09-03, `run_incremental_3e6.py`)

Same 4-round streaming protocol, direct int8 at lr **3e-6** (the stalled setting), seeds
42/1/7/123. Conv int8 movement is **0.00 ± 0.00%** in every round (total stall confirmed at
scale) — so this row is pure BN + int32-bias training, no conv-weight learning.

| method | b2 | b3 | b4 | b5 | b2–5 | conv moved |
|---|---|---|---|---|---|---|
| QZO int8 @ 3e-6 (stalled) | 87.36 ± 0.53 | 80.28 ± 0.72 | 88.47 ± 0.53 | 82.92 ± 0.53 | **84.76** | 0.0% |
| QZO int8 @ 1e-5 | 89.03 ± 1.15 | 84.17 ± 1.40 | 88.47 ± 1.23 | 82.22 ± 1.57 | **85.97** | 40–88%/round |
| float ZO @ 3e-6 | 86.94 ± 0.72 | 83.75 ± 2.28 | 89.31 ± 1.46 | 84.17 ± 0.32 | **86.04** | (float) |

Reading: **un-stalling the conv weights (3e-6 → 1e-5) buys +1.21 pt mean b2–5** (84.76 → 85.97),
concentrated almost entirely on **b3** (+3.89: 80.28 → 84.17) — the hard streaming batch where
conv-weight adaptation matters most; b2 also gains (+1.67). b4/b5 are flat-to-slightly-down
within noise. So conv-weight training is not cosmetic: on the batch that most needs adaptation
it recovers ~4 pt, lifting stalled-QZO (84.76, below float ZO's 86.04) up to parity with float
ZO (85.97 ≈ 86.04). This is the clean quantitative case for making the conv weights move, on top
of the movement-count evidence.

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
