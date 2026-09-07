# exp13 — Findings: train ONLY the fp32 parameters (conv weight + bias frozen), lr 3e-6

2026-09-07 01:05 CEST · Branch `feat/QZO` · Commands in `Plan.md`. Logs: `device_round1_3e6_freeze.log.gz`,
`export_3e6_freeze.log.gz`; params `device_weights_3e6_freeze.npz`, `baked_3e6_freeze_ref/`; 3-way `analysis.txt`.

## Setup
exp12 with the 10 `*_pmul` initializers zeroed in both graphs (`qzo_transform.freeze_conv_pmul`, exporter env
`QZO_FREEZE_CONV=1`): conv int8 weights and int32 biases neither perturb nor update on device or host. Verified: export
reports 10 zeroed initializers per graph; the export's graphs are byte-identical to the hand-frozen device fixture;
`inputs.npz`, node ids and the float-parameter noise are identical to exp12. Device run overlapped with the export;
errors = harness rule (|dev − ref| > 0.001 abs) recomputed from the raw `lp_bits`/`lm_bits` (validated on the 1e-5 log).

## Result (three device-level controls)
| run | int path | errors / 21,600 | first LARGE L+ / L− | final params device vs host |
|---|---|---|---|---|
| lr 1e-5 (exp10) | weights + biases train | **16,565 (76.7%)** | 5 / 326 | all classes differ |
| lr 3e-6 (exp12) | weights frozen, biases train | **4,254 (19.7%)** | 301 / 424 | int8 weights exact; biases 81/104 differ; fp32 ≤2.5e-4 |
| lr 3e-6 frozen (exp13) | **weights + biases frozen** | **5,838 (27.0%)** | 181 / 204 | **10/10 int tensors bit-exact**; 12 fp32 tensors differ, max 1.7e-4 … 6.8e-4 |
exp13 per band (L+): 0–300 99.5% ulp-only; 300–600 4% LARGE; 600–1200 23%; 1200–2700 81% (median rel 1e-2).

## Interpretation
1. **The bias int path is not an amplifier.** Removing it did not reduce the disagreement (it rose; a different chaotic
   realization — the robust statement is that the residual does not need the bias path).
2. **The entire 3e-6 residual is the float path.** With every int parameter bit-exact end-to-end, device and host still
   disagree on 27% of losses. Mechanism: fp32 params (fc, BN γ/β) receive `w ± coeff` where `coeff = f32(−lr·g_proj)`
   differs at a ulp because the fp32 forward tail differs at a ulp (Gemm/BN FMA, picolibc `expf`/`logf`); the difference
   compounds (2e-4–7e-4 by the end); BN drift crosses int8 activation-Quant boundaries → LARGE single-forward
   differences → larger `g` differences → faster drift. The activation Quant is the amplifier inside the float path —
   which is why float ZO (same drift, no Quant) stays within ~1e-4 and QZO does not.
3. **At 1e-5 the int8 weight path adds ×4** (bifurcation at step ~326).

## Consequence for the deliverable (device ↔ host bit-exact carry)
L1/L2 made every host op identical to the device; the update path is verified identical (source + compiled); exp13
shows the int path can be made exactly identical. The only remaining source of difference is the fp32 tail seed.
Everything downstream of it is deterministic on both sides, so removing the seed (L3: `-fno-fast-math -ffp-contract=off`
scoped to the fp32 kernels/graphs — the only combination measured to give fused=0; L4: one deterministic `expf`/`logf` on
both sides) would make `g`, `coeff`, the fp32 parameters and therefore the whole 2700-step carry bit-exact (diff = 0),
not merely within tolerance. Conversely, no change confined to the int update path (freezing, master weights) can reach
float-ZO tolerance while the seed remains: exp13 is "fp32-only training on device" and still diverges to 27%.
Decision pending with the user.

## Micro-trace (2026-09-07) — see `micro/FINDINGS.md`
Layer-by-layer device-vs-host bit-level probes of the step-0 +eps forward: **every integer stage bit-exact** (all five
conv+requant outputs 0 differing elements); **the first difference appears at BN-0** (25% of elements, mostly 1 ulp,
max 4) and is re-created by every BatchNormInternal (25–46%), absorbed by the following Quant, and reaches the loss only
through BN-4 → GAP → fc (logits 7/9 at 1–2 ulp → loss 5 ulp). The int path does not participate. Causal one-kernel
confirmation (strict-fp BatchNorm.c, re-run probe 1 → expect 0) proposed, awaiting go/no-go.

## Reproduction — bit-exact round-1 on device with the fixes applied (2026-09-07)

Repo state: Onnx4Deeploy ≥ `e3aa35f` (host mirrors in `onnx4deeploy/utils/onnx_node_implementations.py`, `QZO_FREEZE_CONV`
in the exporter, `qzo_transform.freeze_conv_pmul`), TrainDeeploy ≥ `4674b3e` (strict-fp32 CMake options) + `1a17208`
(harness `OUTPUT_TOL`/`OUTPUT_BITS`, only needed for the layer probes). Containers: `agitated_hugle` (Onnx4Deeploy at
`/app/Onnx4Deeploy`, TrainDeeploy at `/app/TrainDeeploy`, SilentWear at `/app/SilentWear`), `traindeeploy` (the ETH tree at
`/app/ETH/...`, device toolchain + GVSoC). Setting: lr 3e-6, eps 0.01, n_accum 4, seed 42, 2700 steps, 54 windows,
pooled@99.99 thresholds, frozen-stats BN, conv weight+bias frozen (only the 12 fp32 params train).

**1. Host reference + fixture (agitated_hugle, ~4.5 h; already produced: `exp13/baked_3e6_freeze_ref/`)**
```bash
docker exec -d agitated_hugle bash -c "cd /app/Onnx4Deeploy && QZO_FREEZE_CONV=1 \
  QZO_POOLED_THRESHOLDS=/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/fixture/pooled_9999_fold3.json \
  python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher \
    --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
    --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
    --subject S01 --session 3 --condition vocalized --batch 1 --data-size 54 --stratified \
    --n-epochs 200 --n-accum 4 --lr 3e-6 --bn-frozen-stats \
    -o /app/Onnx4Deeploy/QZO_exp/exp13/baked_3e6_freeze_ref > /app/Onnx4Deeploy/QZO_exp/exp13/export_3e6_freeze.log 2>&1"
```
Check: the log shows `[QZO_FREEZE_CONV] zeroed 10 *_pmul initializers` for both graphs and
`QZO multi-step sim: data_size=54 n_accum=4 → n_steps=2700 lr=3e-06 eps=0.01 seed=42`.
(For the un-frozen or 1e-5 variants drop `QZO_FREEZE_CONV=1` / change `--lr`; the graphs and `inputs.npz` are lr-independent.)

**2. Optional 10-second smoke (host step-0 bits)**: `docker exec agitated_hugle python3 /app/Onnx4Deeploy/QZO_exp/exp13/check_step0_freeze.py`
→ `3eb41bcd 3f09326b 3d4011cf 3efd62eb`; the device's first four `lp_bits` must match these with the strict build.

**3. Pack the device fixture (traindeeploy)** — from the export dir itself, so its `outputs.npz` is the compiled-in reference
and the harness's printed `Errors:` line is the result directly (my run packed from `baked_3e6_freeze`, a byte-identical
graph copy with a placeholder reference, and used the recount below instead):
```bash
docker exec traindeeploy bash -c 'cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/QZO_exp/exp13/baked_3e6_freeze_ref \
    /app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet speechnet_qzo_lr3e6_freeze_train speechnet_qzo_lr3e6_freeze_update'
```

**4. Device round-1 with the strict-fp32 build (traindeeploy, ~3.5 h)** — kill orphan GVSoC by PID first:
```bash
pgrep -f "[g]vsoc_launcher" | xargs kill -9
docker exec -d traindeeploy bash -c 'cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 deeployMezoRunner_tiled_siracusa.py \
    -t Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_freeze_train \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_freeze_update \
    --n-steps 2700 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --cores 8 \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON DEEPLOY_STRICT_FP32=ON \
       "DEEPLOY_STRICT_FP32_FILES=BatchNorm.c;Gemm.c;GlobalAveragePool.c;RandomNoise.c" \
    > /app/ETH/Onnx4Deeploy/QZO_exp/exp13/micro/device_round1_3e6_freeze_strict.log 2>&1'
```
Check in the log: 5 `[QW strict-fp32]` lines (4 kernels + generated graphs), `[BN_FROZEN_STATS]`, 10800 `lp_bits` + 10800 `lm_bits`,
22 `[WDUMP ...]` lines. Without the two strict `-D`s the same command reproduces the fast-math result (5,838 errors).

**5. Verify**
```bash
# (a) losses: harness rule + per-band bit-level breakdown (host-side)
python3 QZO_exp/exp13/verify_bitexact.py QZO_exp/exp13/micro/device_round1_3e6_freeze_strict.log[.gz] QZO_exp/exp13/baked_3e6_freeze_ref/outputs.npz
# (b) final parameters: bit-compare the device dump against the host's updated_* tensors
python3 TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/extract_qzo_weights.py \
  --gvsoc-log <log> --train-onnx QZO_exp/exp13/baked_3e6_freeze_ref/network_zo_train.onnx \
  --out /tmp/dev_w.npz --ref-outputs QZO_exp/exp13/baked_3e6_freeze_ref/outputs.npz
```
Expected: `recount (harness rule): 0 out of 21600`, first LARGE step `None`, 0.0% LARGE in every band, median rel
2.8e-8 … 6.6e-8 (≤ 1 ulp), 5–24% bit-exact per band; `15 bit-exact, 7 differ` with max 1.49e-8 / 2.24e-8 (= 1 ulp) on
BN β and fc. Actual run: `micro/strict_round1_analysis.txt`.
