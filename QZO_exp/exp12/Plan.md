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

## Reproduction — exact commands as executed (for checking my execution)

Containers: `agitated_hugle` = Onnx4Deeploy/ORT/Brevitas (repo at `/app/Onnx4Deeploy`, TrainDeeploy at
`/app/TrainDeeploy`, SilentWear at `/app/SilentWear`); `traindeeploy` = device toolchain + GVSoC (ETH tree at
`/app/ETH/...`). Orphan GVSoC is killed by PID on the host before every device run.

**1. Host reference export at lr 3e-6** (identical to the 1e-5 export except `--lr`; ~4.5 h):
```bash
docker exec -d agitated_hugle bash -c "cd /app/Onnx4Deeploy && \
  QZO_POOLED_THRESHOLDS=/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/fixture/pooled_9999_fold3.json \
  python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher \
    --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
    --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
    --subject S01 --session 3 --condition vocalized --batch 1 --data-size 54 --stratified \
    --n-epochs 200 --n-accum 4 --lr 3e-6 --bn-frozen-stats \
    -o /app/Onnx4Deeploy/QZO_exp/exp12/baked_3e6 > /app/Onnx4Deeploy/QZO_exp/exp12/export_3e6.log 2>&1"
```
Check: `grep "multi-step sim" export_3e6.log` must show `data_size=54 n_accum=4 → n_steps=2700 lr=3e-06 eps=0.01 seed=42`;
`baked_3e6/network_zo_train.onnx`, `network_zo_update.onnx` and the params in `inputs.npz` must be identical to
`exp10/baked_200ep/` (lr is not baked into the fixture; the update graph carries `eps=0.01`).

**2. Smoke (chain validation, 2 steps, fixture packed from the lr-independent 1e-5 export):**
```bash
pgrep -f "[g]vsoc_launcher" | xargs kill -9
docker exec traindeeploy bash -c 'cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/QZO_exp/exp10/baked_200ep \
    /app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet speechnet_qzo_lr3e6_train speechnet_qzo_lr3e6_update'
docker exec -d traindeeploy bash -c 'cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 deeployMezoRunner_tiled_siracusa.py -t Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_train \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_update \
    --n-steps 2 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --cores 8 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON \
    > /app/ETH/Onnx4Deeploy/QZO_exp/exp12/smoke_3e6_2step.log 2>&1'
```
Check: `[BN_FROZEN_STATS]` line present; `[WDUMP ...]` lines present; step-0 `lp_bits` == the 1e-5 run's step-0
bits `3f9d48ea 3e2d5aa0 3dffdca0 3fc556fd` (the forward is lr-independent).
Note: `--dump-weights` is NOT a runner flag (first attempt failed on argparse); the dump is the CMake passthrough
`-D DUMP_WEIGHTS=ON`, same path as `BN_FROZEN_STATS`.

**3. Full device round-1 at 3e-6** (after step 1 finishes; re-packs from `baked_3e6` so its `outputs.npz` is the
compiled-in reference the harness compares against; ~5 h): `./run_device_3e6.sh` (this dir; same runner command
with `--n-steps 2700`, log → `device_round1_3e6.log`).

**4. Analysis:** `python3 analyze_exp12.py` — prints the harness `Errors: N out of 21600` line, the same count
recomputed from `lp_bits`/`lm_bits` with the harness rule (|dev − ref| > 0.001 abs; validated on the 1e-5 log:
8302 + 8263 = 16565 exactly), per-band categories for L+ and L−, and the first LARGE step — for both runs side by side.

**Smoke result (2026-09-06 17:27):** `[BN_FROZEN_STATS]` on, 22 `[WDUMP]` lines, step-0 `lp_bits` == 1e-5 run
(`3f9d48ea 3e2d5aa0 3dffdca0 3fc556fd`) → chain and lr-independence validated on device. The harness line
`FAILED - 6 errors out of 16` is EXPECTED here: the smoke was compared against the placeholder 1e-5 reference
(8 step-0 losses match, the step-1 losses differ because the step-0 update used a different lr). Not a defect.

**Overlap decision (17:40):** the device round-1 was launched BEFORE the 3e-6 export finished, using the already-packed
lr-independent fixture (its compiled-in reference is the 1e-5 placeholder). Rationale: the device run does not need the
3e-6 reference to execute, and the harness rule recomputed from the logged raw `lp_bits`/`lm_bits` reproduces the
harness count exactly (validated: 8302 + 8263 = 16565 on the 1e-5 log). Therefore the reported exp12 `Errors:` count is
`analyze_exp12.py`'s recount of the device bits against `baked_3e6/outputs.npz`; the harness's own printed line in
`device_round1_3e6.log` is against the placeholder and must be ignored. Saves ~5 h wall.
Launch (host): `pgrep -f "[g]vsoc_launcher" | xargs kill -9`, then the same runner command as the smoke with
`--n-steps 2700` → `device_round1_3e6.log` (no re-pack).
