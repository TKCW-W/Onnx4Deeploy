# exp3 — lr 1e-5 direct-int8 QZO: cross-fold stability vs float ZO (Plan)

Date: 2026-09-03 · Branch `feat/QZO` · Container `agitated_hugle` (Brevitas 0.13.0)

## Question

Is the lr = 1e-5 direct-int8 quantized-ZO setting **stable across folds**? Until now the whole
QZO=float-ZO result (`../exp_calibration/`) was on **fold 3 only**. Here we repeat the incremental
fine-tuning on **folds 1 and 2** and check that QZO still reaches accuracies **similar to float
ZO**, per fold. We also document the **fraction of conv weights updated at each FT round**, to
confirm the strong-signal-filter mechanism is not fold-specific.

## Fold definition (verified)

`leave_one_session_out_fold_k.pt` holds out **session k** (confirmed by zero-shot: the held-out
session is the lowest — fold1 sess1=73.3, fold2 sess2=63.9, fold3 sess3=80.6). So for fold k:
- **pretraining data** (for the checkpoint AND our PTQ calibration) = the OTHER two sessions.
- **fine-tuning + eval data** = the held-out session k, batches 1..5.

Checkpoints (S01/vocalized/speechnet/w1400ms/model_1/): `leave_one_session_out_fold_{1,2,3}.pt`.

## Protocol (identical to exp_calibration §4d, per fold)

Streaming incremental, rounds r = 1..4: fine-tune on **54 stratified windows** (30% of 180,
6/class, seed 42) from session-k **batch r**, carry weights, evaluate on the WHOLE session-k
**batch r+1** (180 windows). b1 is only ever a training batch (never evaluated), so the metric is
mean over eval batches b2..b5. Recipe: 200 epochs × 54 / n_accum 4 = **2700 update steps/round**,
ε 0.01, z-seed restarts at 42 each round.

Two pipelines on the SAME per-round fixtures:
- **QZO**: direct int8, **lr 1e-5**, activation scales frozen at **pooled@99.99** (calibrated on
  THIS fold's pretraining sessions, 1800 windows), weight scales at pretrained per-channel
  abs-max — both frozen across all rounds (no recalibration mid-stream).
- **float ZO**: plain SpeechNet, **lr 3e-6** (its recipe), eval-mode BN, all 22 tensors trainable.

Per-fold, per-round we record: eval-batch accuracy before/after, and QZO conv int8 movement
(this-round net, cumulative net vs pretrained, ever-moved union), out of 14,880 conv weights.

## Per-fold calibration (important — it is fold-specific)

The pooled@99.99 activation calibration is recomputed on each fold's pretraining distribution
(the two non-held-out sessions, 10 batches × 180 = 1800 windows), streamed batch 64, pooled |x|
per site, one 99.99 quantile, frozen (ConstThreshold, bit-exact). Weight scales are data-free
abs-max from that fold's checkpoint. Nothing is inherited from the fold-3 study.

## Metrics / success criterion

- Primary: per-fold mean b2–5 balanced accuracy, QZO vs float ZO. "Stable" ⇒ QZO within ~1 pt of
  float ZO on each fold (the fold-3 gap was 0.07 pt over 4 seeds), and both improve over zero-shot.
- Secondary: per-round conv-weight-moved fraction is non-trivial every round on every fold (the
  filter keeps firing), matching fold-3 (40–88%/round).

## Reproduction guide

All in `agitated_hugle`, from `/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability`:

```bash
# runs folds 1 and 2 (fold 3 numbers are taken from ../exp_calibration for the summary):
python3 run_all.py                 # calibrate per fold -> QZO + float ZO incremental -> results.json
# single fold, foreground:
python3 -c "import run_all; run_all.run_one_fold(1)"
```

Depends on `../exp_calibration/{run_study.py, run_incremental.py, calib_pooled.py}` (added to
`sys.path`); reuses their generic ZO machinery (build_params, run_regime, install, draw_z,
balanced, logits_of, pooled calibration). Fold-specific pieces (checkpoint, per-fold model/data,
per-fold calibration) live in `stability_lib.py`. Data: SilentWear S01/vocalized under
`/app/SilentWear/SilentWear_data/data_raw_and_filt`. Outputs: `results.json` (incremental,
resumable — completed folds skipped), `run.log`, `Findings.md`.

Seed: primary run at seed 42 (matches the fold-3 headline). If per-fold single-seed variance
looks large, extend to seeds {1,7} (the runner accepts a seed list).

## Deliverables

`Plan.md` (this), `stability_lib.py`, `run_all.py`, `results.json`, `run.log`, `Findings.md`
(dated; per-fold QZO-vs-float table + per-round movement table + cross-fold verdict).
