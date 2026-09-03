# exp3 — lr 1e-5 direct-int8 QZO cross-fold stability — Findings

Date: **2026-09-03** · Branch `feat/QZO` · `agitated_hugle` · seed 42 · Raw: `results.json`, `run.log`

## One-line result

**lr 1e-5 direct-int8 QZO is NOT uniformly stable across folds.** It matches float ZO on folds 1
and 3 but underperforms by ~4.9 pt on fold 2 — the hardest held-out session — because a single
fixed lr cannot place the rounding threshold correctly for every fold's gradient scale.

## Protocol

Per fold k (holds out session k; pretraining = the other two sessions — verified by zero-shot:
fold1 sess1=73.3, fold2 sess2=63.9, fold3 sess3=80.6 lowest). Streaming incremental rounds
r=1..4: FT on 54 stratified windows (30% of 180, 6/class, seed 42) of session-k batch r → eval
whole batch r+1 (180 windows). 200 ep × 54 / n_accum 4 = 2700 steps, ε 0.01, z-seed 42/round.
QZO = direct int8, lr 1e-5, act scales pooled@99.99 (calibrated per-fold on that fold's 1800
pretraining windows), weight scales abs-max, all frozen across rounds. float ZO = lr 3e-6, all
22 tensors trainable, eval-BN. Same per-round fixtures for both. Fold 3 taken from
`../exp_calibration` (same protocol/seed). Pipeline validated: exp3 reproduces fold-3 zero-shot
to within 1 window (85.00 vs 85.56).

## Results (mean b2–5 balanced accuracy)

| fold | held-out sess | QZO @1e-5 | float ZO @3e-6 | **QZO − float** | conv moved/round | cum-net / union |
|---|---|---|---|---|---|---|
| 1 | 1 | 88.89 | 88.61 | **+0.28** | 89/73/90/66% | 89.6% / 99.8% |
| 2 | 2 | 77.08 | 81.94 | **−4.86** | 90/93/62/94% | 95.7% / 99.9% |
| 3 | 3 | 86.11 | 86.11 | **+0.00** | 66/43/89/41% | 85.8% / 98.3% |

Per-batch (QZO / float, b2..b5):
- fold 1: QZO [88.3, 92.2, 92.8, 82.2] / float [90.0, 91.1, 92.2, 81.1]
- fold 2: QZO [77.2, 78.9, 76.7, 75.6] / float [81.7, 83.3, 80.6, 82.2]
- fold 3: QZO [90.0, 84.4, 87.8, 82.2] / float [87.8, 83.3, 89.4, 83.9]

## Reading

- **Folds 1 & 3: QZO ≈ float ZO** (+0.28, +0.00) — the lr-1e-5 setting reproduces float-ZO
  accuracy, as on the original fold-3 study.
- **Fold 2: QZO −4.86 below float, and below on EVERY batch** — a systematic loss, not one bad
  eval. Fold 2 is the hardest fold (held-out session-2 zero-shot 63.9%), and it shows the
  **highest** conv movement (90%+ in 3 of 4 rounds, cum-net 95.7%). That combination — most
  movement, worst accuracy — is the signature of the over-movement / random-walk regime
  (cf. exp_calibration §4c, where lr≥3e-5 on fold 3 gave 97–99% movement and collapsing accuracy).
  Fold 2's larger gradients push the fixed lr-1e-5 threshold too low, so weak-signal steps also
  clear it and the int8 weights random-walk rather than track the gradient.
- **This is the per-bench-lr caveat, demonstrated.** The magic of 1e-5 is that it places
  `0.5·s_w/lr` at the *tail* of a fold's |g| distribution. That placement was tuned on fold 3;
  fold 2's distribution is shifted, so the same lr lands in the wrong regime. A fixed lr is
  therefore fragile across folds.

## Conconclusion & recommendation

- lr 1e-5 direct-int8 QZO is a **valid but not robust** setting: it works where its threshold
  happens to sit at the |g| tail (folds 1, 3) and degrades where it does not (fold 2).
- **Per-fold lr tuning** would likely recover fold 2 (a lower lr, e.g. 3–6e-6, to raise the
  threshold): the principled rule from exp_calibration — place `0.5·s_w/lr` at the top ~1% of a
  100-probe |g| sample — is measurable per fold. Untested here; the clean next step.
- The **lr-insensitive** alternatives (master weights, error-feedback residual) remain the robust
  option if a single fixed deployment lr is required across subjects/folds — they preserve
  sub-LSB signal instead of relying on threshold placement, so they do not have this fragility.

## Caveats

Single seed (42) per fold — fold 2's −4.86 is consistent across all 4 batches (systematic-
looking) but a 2–3-seed confirmation on fold 2 would rule out a seed-unlucky |g| stream. Single
subject (S01), host sim, 180-window eval (1 window = 0.56%). float ZO at its recipe lr 3e-6 vs
QZO's fold-3-tuned 1e-5 (not iso-lr).

## Reproduction

See `Plan.md`. In `agitated_hugle`, `/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability`:
`python3 run_all.py` (folds 1,2; fold 3 pulled from ../exp_calibration). Resumable.
