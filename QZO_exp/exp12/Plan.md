# exp12 — control: device vs host-ref at lr 3e-6 (int8 update inert)

Started 2026-09-06 · Branch `feat/QZO` · Proposed by the user as a control for exp11.

## Hypothesis to separate
The 1e-5 device round-1 disagrees with its host reference on **16565 / 21600** losses (device
harness `Errors:` line = |device − ref| > TOL 0.001 abs over 10800 L+ and 10800 L−). Is that
attributable to the **float part** of the path (fp32 tail: GAP/fc/SCE + fc/BN float-param
updates, plus device-side FMA/libm) or to the **int8 update path**?

## Design
Identical pipeline to the 1e-5 run — same fixture (the export is lr-independent: graphs + inputs.npz
are byte-identical, only the host-reference `outputs.npz` changes), same seed 42, data (54 windows),
pooled@99.99 calibration, n_accum 4, 2700 steps, frozen-BN — with **lr = 3e-6**, where (calibration
study) the direct-int8 conv weights **never move** (0.00% changed). The int8 update path is then inert;
only the float parameters (fc, BN) still update, and the fp32 forward runs unchanged.

| outcome | reading |
|---|---|
| `Errors:` count ≈ 16565 (does not change much) | disagreement lives in the **float part** (forward fp32 / activation flips / float-param drift), not the int update |
| `Errors:` count drops a lot | the **int8 update path** is implicated |
Secondary readout: if the LARGE-event rate is lr-independent → forward activation flips; if it scales
with lr → float-parameter drift.

## Prediction (from exp11 evidence, stated before the run)
Count will NOT change much: (a) the direct-int8 update is robust to a ~3e-7 seed (surrogate: 600
steps, 0/14984 weights differing); (b) the 1e-5 LARGE events start at step 5 with identical weights
and are transient; (c) aligning every host op we own changed nothing.

## Steps / artifacts (this dir)
1. `export_3e6.log`, `baked_3e6/` — host reference at 3e-6 (`Onnx4Deeploy.py -mode q-zo-train ... --lr 3e-6`).
2. `smoke_3e6_2step.log` — pack→build→run chain validated on 2 steps first; step-0 `lp_bits` must equal the 1e-5 run's (forward is lr-independent).
3. `device_round1_3e6.log[.gz]` — full 2700-step GVSoC run (`deeployMezoRunner_tiled_siracusa.py ... --lr 3e-6 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`).
4. `analysis/` — `Errors:` line from the log; per-band LARGE table vs the 1e-5 run; final-weight diff (int8 must be identical; float drift isolated).
