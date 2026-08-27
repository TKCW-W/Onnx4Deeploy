# Plan — Quantized ZO (QZO) as a true extension of the float ZO + shipped quantized pipelines

**Repo/branch:** `Onnx4Deeploy` + `TrainDeeploy` on `feat/QZO`.
**Date started:** 2026-08-27. **Driver:** self-paced `/loop`.

## Decision (locked this run)
- **Weights quantized OFFLINE (standard).** int8 weight *initializers* (values in [-127,127]) with the
  per-channel weight scale folded into the **per-channel RequantShift** `mul[c]=s_in·s_w[c]/s_out`. This is
  the shipped Deeploy datapath (`RequantShiftTemplate` general per-channel case; `UniformRequantShift` is the
  per-tensor optimization). **No runtime weight-Quant kernel.**
- **Online fp32-master weight perturbation is DEFERRED** as a later optimization direction (it needs a
  per-channel *Quant* kernel Deeploy doesn't ship; a few days of Deeploy work). Not in scope this run.
- **Retire the from-model rebuilder** `build_qzo_int8_graph` (it bypassed the export and hand-rebuilt the
  int8 graph → wrong architecture, JSON side-channel, `blocks.0.conv.weight` naming). Comment out, do not
  delete (shipped-file policy).

## Target design — the true extension
Transform the **shipped quantized INTEGER graph** (the DeepQuant→Deeploy-frontend output, "network.onnx":
int8 conv weight initializers, per-channel RequantShift, Quant/Dequant, unfolded fp32 BatchNormalization)
through the SAME `inject_perturbation_nodes` machinery the float ZO path already uses. Key realization:
`inject_perturbation_nodes(noise_type="rqs_rademacher")` **already**:
- perturbs Conv/Gemm int8 weights with `RQSPerturbRademacher` (int8, div=2¹⁵, n_levels=256),
- perturbs the RequantShift int32 bias (`add`, input idx 2) with `RQSPerturbRademacher` (div from node, 2³²),
- re-emits `BatchNormalization → BatchNormInternal` (com.microsoft, training_mode=1, 5 outputs),
- **promotes perturbed trainable initializers → graph INPUTS** (`_promote_initializers_to_inputs`),
- appends the canonical `SoftmaxCrossEntropyLoss` (`append_cross_entropy_loss`).

**What's missing (the actual code delta):** per-node MIXED noise. SpeechNet-quant needs int8 `rqs_rademacher`
on Conv/Gemm weights + RequantShift bias, but **float `rademacher`** on the fp32 BatchNormalization γ/β.
Today `inject_perturbation_nodes` applies ONE global `noise_type`. Extend it to pick the perturb op by the
target tensor's dtype/producer: int8/int32 tensor → RQS variant; float tensor (BN γ/β) → plain float variant.

Result: trainable params (int8 conv/fc weights, int32 RequantShift biases, fp32 BN γ/β) are **graph INPUTS**,
BN is `BatchNormInternal`, weights are offline-int8 with per-channel RequantShift — exactly the shipped
deployable form, produced by TRANSFORMING the export (not rebuilding).

## Steps
1. **[study]** Map the shipped integer-graph export (DeepQuant `exportBrevitas` + Deeploy frontend passes);
   confirm the artifact shipped to TrainDeeploy is the *integer* graph (agent in flight). ← gating fact
2. **[export base]** Produce the SpeechNet quantized INTEGER inference graph from `QuantSpeechNetDeploy`
   calibrated on **real** SilentWear data (S01/session3/vocalized) + fold_3 pretrained weights.
3. **[transform]** Extend `inject_perturbation_nodes` for mixed per-tensor noise; run it on the integer graph
   → `zo_train` (weights-as-inputs, BatchNormInternal, int8 RQS + float BN perturb) + SCE loss.
4. **[update]** Extend `generate_weight_update_graph` the same mixed way → `zo_update`.
5. **[wire]** Route `_export_qzo_training` through this transform; keep the `q-zo-train` CLI mode.
6. **[reference]** Host reference L+/L− via ORT (ensure `run_onnx_graph` has Quant/Dequant impls).
7. **[artifacts]** `QZO_exp/exp2/`: fixture (`network_zo_train.onnx`, `_zo_update.onnx`), `inputs.npz`,
   `outputs.npz`, scales, generate script. Real data + pretrained weights, correct paths.

## Optional (only if 1–7 succeed) — TrainDeeploy stage
8. Pack the fixture (`experiments/zo_smoke/pack_2step_fixture.py`) → `Tests/Models/Training/SpeechNet/
   speechnet_qzo_{train,update}`.
9. Single-step on-device sim via `deeployMezoRunner_tiled_siracusa.py` (extend if needed), `-D
   BN_FROZEN_STATS=ON`, n_steps 1, n_accum 1, num-data-inputs 2, eps 0.01. Kill orphan gvsoc by PID first.
10. Compare device L+/L− vs host reference. **Verify device kernels use INT8 conv** (grep generated
    TrainingNetwork.c for pulp-nn int8 conv, not `PULP_Conv2d_Im2Col_fp32`).

## Refinement after iteration-1 study (2026-08-27)
The real `-mode quant` integer graph + shipped q-zo wiring changed two details (see Report.md §Iteration 1):
- **Base graph = `-mode quant` `network.onnx`** (int8 input, activation Quant/Dequant/RequantShift). But its
  **conv/fc weights stay f32 behind a QCDQ chain** (`Div→Add→Round→Cast→Cast→Clip→Sub→Mul→Conv`) — the `Cast×2`
  break constfold. Must **constfold each weight/bias QCDQ → int-valued initializer feeding the consumer
  directly** (option **B**, localized to the QZO path) so it matches the shipped q-zo, where the conv reads
  `weight_init → RQSPerturbRademacher → Conv` directly.
- **BN is folded** by the quant export ⇒ trainable params = 12 (5 conv w+b, fc w+b), **no BN γ/β**, so **no
  mixed float/RQS noise** — every param is int-valued → all `RQSPerturbRademacher`. BN-unfolded γ/β training
  is a follow-up. `inject_perturbation_nodes(rqs_rademacher)` then perturbs the (now-direct) weight/bias inits
  and `_promote_initializers_to_inputs` makes them graph INPUTS.
- **Export env:** `agitated_hugle` needs `PYTHONPATH=/app/Onnx4Deeploy/DeepQuant` (DeepQuant not pip-installed).

## Paths (verified on host 2026-08-27)
- Host `/Users/qiwenwu/ETH` → `agitated_hugle:/app`, `traindeeploy:/app/ETH`.
- Checkpoint: `/Users/qiwenwu/ETH/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
  (container `agitated_hugle`: `/app/SilentWear/...`).
- Data: `/Users/qiwenwu/ETH/SilentWear/SilentWear_data/data_raw_and_filt` (container `/app/SilentWear/...`).

## Guardrails
Conservative; no faked results; comment-out don't-delete for shipped SpeechNet files; kill orphan gvsoc by
explicit PID before every sim; commit milestones with the Co-Authored-By trailer. Exports run in
`agitated_hugle` (has onnxruntime-training + DeepQuant); sims in `traindeeploy`.
