# exp_calibration — Is the calibration behind the LSB stall? (Plan)

Date: 2026-09-02 · Branch: `feat/QZO` · Runs in container `agitated_hugle` (Brevitas 0.13.0)

## Question

Re-examine, **without presuming the earlier refutations**, whether the activation-scale
calibration is (part of) the cause of the LSB stall in direct-int8 quantized ZO — and, no matter
what the answer is, leave behind a **correctly implemented calibration procedure** at the
implementation level.

## Why re-open this

Our current calibration (ours, not inherited — the shipped `Onnx4Deeploy_ZO`
`export_zo_training(quant=True)` path calibrates nothing; it exports uncalibrated default scales
0.007812 from `torch.randn` input) has three known weaknesses:

1. **Tiny sample**: 8 windows (later 54), drawn from *session 3 batch 1* — the target session,
   not the pretraining distribution the checkpoint was trained on.
2. **Degenerate percentile**: Brevitas `Int8ActPerTensorFloat` uses `AbsPercentile` @ 99.999,
   computed **per forward call** on the current batch. For a batch of 1 window the largest site
   has 9 800 values → the 99.999th percentile is literally the max; even at batch 64
   (627 200 values) it is the ~6th-largest value. So in practice we were doing abs-max
   calibration, outlier-inflated by construction.
3. **EMA, not pooling**: `ParameterFromRuntimeStatsScaling` EMA-averages the per-batch
   percentiles (momentum 0.1). The result depends on the batching schedule and ordering — it is
   an estimator of "the average per-batch near-max", not a property of the data.

### Evidence trail (traceable)

Our production calibration call — `onnx4deeploy/core/base_exporter.py:706`:

```python
with torch.no_grad(), calibration_mode(model):
    model(calib)
```

Brevitas collector (per-call percentile → running stats), Brevitas 0.13.0
`brevitas/core/stats/stats_op.py` (`AbsPercentile.forward`):

```python
def forward(self, x: Tensor):
    ...
    # k-th largest of |x| for THIS call only; pooling across calls happens outside,
    # in ParameterFromRuntimeStatsScaling, as an EMA over these per-call values
    result = x.abs().view(-1).kthvalue(k).values
```

and `brevitas/core/scaling/runtime.py` (`ParameterFromRuntimeStatsScaling`): buffer updated with
momentum 0.1 for `collect_stats_steps` calls, then frozen into a learned parameter.

Shipped path exports *uncalibrated* — `Onnx4Deeploy_ZO/onnx4deeploy/core/base_exporter.py:539`
(`export_zo_training`): `model.eval()` + `input_tensor = torch.randn(*input_shape)` +
`exportBrevitas(model, input_tensor)`; `quant_inference_mode` never runs the stats collector
(collection is gated on `training` mode). Empirically: every act scale = 0.007812 = 1/128
(uninitialised default), uniform across all layers.

## The corrected calibration procedure (what "proper" means here)

- **Data**: the **pretraining distribution** — S01 / vocalized / sessions 1+2, all 10 batches
  × 180 = **1800 windows** (fold_3 protocol: the checkpoint was trained on sessions 1+2;
  session 3 is held out). Verified on disk: `data_raw_and_filt/S01/vocalized/sess_{1,2}_batch_{1..5}.h5`.
- **Collector**: **pooled**, not EMA. Stream batches of 64 (1800/64 → **29 batches**: 28 full +
  one of 8), hook the input of every activation quantizer, pool the |x| populations across all
  batches, and take **one** quantile per site at the end. This is a pure data property —
  independent of batch size, ordering, and momentum.
- **Percentile**: swept — see below. Note that pooling *rescues* the high percentiles: at the
  largest site the pooled population is 1800 × 9 800 ≈ 17.6 M values, so the 99.999th percentile
  is the ~176th-largest value — a genuine tail statistic, no longer the max. We record, per site
  and per setting, the effective rank so this stays auditable.
- **Observation mode**: identical to what `calibration_mode` observes — float activations
  (quantizers bypassed during collection), pretrained weights loaded, `eval()` BN. We collect via
  forward-pre-hooks on each act-quant proxy so the pooled procedure sees exactly the tensors the
  Brevitas collector would see.
- **Freezing**: computed threshold `q_p` written into the quantizer (Brevitas act scale
  convention: `s = q_p / 128` for non-narrow Int8ActPerTensorFloat) and **frozen**
  (ConstScale surgery, same machinery validated in `exp_brevitas_stall`). Weight scales stay
  data-free per-channel abs-max (`max|W[c]|/127`) — calibration does not, and cannot, touch the
  weight grid; this is stated up front because the stall lives in the weight grid.

## Sweep

| axis | values |
|---|---|
| percentile | 99.9 · 99.99 · 99.999 · 100 (abs-max) |
| collector | pooled-1800 (primary) · current EMA/54-window session-3 (baseline for comparison) |

Plus two fixed reference points: the shipped-style uncalibrated scales (0.007812 everywhere) and
the float model.

## Metrics — "calibration quality"

Per configuration, on the held-out session 3:

1. **Zero-shot quantized balanced accuracy** on batch 2 (whole batch, 180 windows) — primary.
2. **Forward fidelity**: logit cosine vs the float model + mean |Δ cross-entropy| on batch 2.
3. **Per-site health**: saturation rate (fraction of |x| clipped at 127·s) and zero-bin rate
   (fraction quantised to 0) — the two failure modes of a wrong scale.

## Stall linkage — the actual question (measured, not assumed)

With the *best* calibration and with the *old* calibration, on the FT protocol
(54 stratified windows, session 3 batch 1, ε=0.01, lr as in the FT study):

4. Distribution of the ZO probe signal |L₊ − L₋| and of g = (L₊−L₋)/(2ε) over ≥100 probe steps
   (shared seeds across configs).
5. Implied per-channel update in LSB units: |lr · g / s_w[c]| — and the fraction of steps
   clearing the 0.5 LSB rounding threshold. This is the direct stall criterion.
6. Short direct-int8 fine-tune (Regime B, ~100–300 steps) under best-vs-old calibration:
   does the int8 weight move at all, and what does batch-2 accuracy do? Regime A
   (master-weight) under the same calibrations as control.

Critical framing: activation calibration can change g (through forward fidelity and loss
sharpness) but not the LSB size (s_w is data-free). The open empirical question is whether a
properly calibrated forward makes |g| systematically larger (sharper loss differences) by enough
to clear 0.5 LSB. We measure that; we do not assume either answer.

## Deliverables

```
QZO_exp/exp_calibration/
  Plan.md                (this file)
  calib_pooled.py        (the pooled-percentile calibration implementation, reusable)
  run_study.py           (sweep + metrics + stall linkage)
  results.json           (all numbers)
  *.png                  (scale comparison, quality metrics, LSB-clearance distribution)
  Findings.md            (dated summary; written after the runs)
```

Follow-up (separate commit, after Findings): port the winning procedure into the production
export path (`base_exporter.py` calibration site), comment-out-don't-delete the old call.

## Answers to the framing questions (short form)

- *"1800/64 → 29?"* Yes: 28 full batches of 64 + one of 8 = 29 collector steps to see all data.
  (With true pooling the step count only affects memory, not the estimate.)
- *"Is 99.999 basically the max?"* Per-batch, essentially yes (rank ≈ 6 of 627 k at batch 64;
  literally the max at batch 1 or at small sites). Pooled over 1800 windows, no —
  rank ≈ 176 of 17.6 M at the largest site. Pooling is what makes the percentile axis meaningful.
- *"EMA vs pooling?"* Pooling. EMA is an artefact of Brevitas' streaming implementation; it
  weights recent batches ~exponentially and so encodes the batching schedule. A scale should be
  a statistic of the data, so: pool, then one quantile.
