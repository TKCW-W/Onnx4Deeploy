<!-- SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna ; SPDX-License-Identifier: MIT -->
# 2026-08-04 — PLAN: port the ZO implementation + add SpeechNet ZO on `feat/ZO`

**Branch:** `feat/ZO`, created from **`feat/BNFRozen_OptionB`** (our SpeechNet branch: SpeechNet model +
exporter, `SilentWearDataSource`, windowing, BN-frozen + single-step training, CLI already registers
SpeechNet and the `--dataset/--bn-frozen/--stratified/--pretrained-weights` flags).
**Source of ZO code:** the `ZO` branch (colleague's MeZO impl, commit `147285e`) studied in detail (see
`demo_bp_vs_zo/TUTORIAL_BP_vs_ZO.md` + `zo_rademacher/ZO_PAPER_STUDY.md` on `ZO`).
**Goal:** `feat/ZO = feat/BNFRozen_OptionB + ZO implementation + SpeechNet-ZO additions`, where SpeechNet
**trains its BN affine (γ/β) via ZO but keeps BN running stats frozen** for fine-tuning and inference.

> **This is a plan only — no code ported yet.** Scope = the **Onnx4Deeploy** side (generate correct SpeechNet
> ZO graphs + a bit-faithful reference loss). The Deeploy/TrainDeeploy kernel + runner side is out of scope
> (see §6).

---

## 1. Faithful-algorithm context (what the graphs must support)
From the ZO paper (Alg. 1) + exp18: the deployed ZO update is **one shared direction `z` per accumulation
window, scalar accumulation of `(L₊−L₋)`, one in-place update**, optionally `q` directions. The **graph** side
(this task) only needs to emit:
- `network_zo_train.onnx` — perturbed forward + CE loss (one seed → one scalar loss), and
- `network_zo_update.onnx` — per-weight in-place perturbation (the update step).
The shared-`z`/scalar-accumulation/`q`-loop lives in the **runner** (Deeploy), not the graph. So the port is
graph-generation + Python reference; correctness of the estimator loop is a runner concern.

## 2. Current state (verified)
| | `feat/ZO` base (=BNFRozen) | `ZO` branch |
|---|---|---|
| SpeechNet exporter/model/datasource/windowing | ✅ | stale `.pyc` only |
| BN-frozen + `_fold_bn_into_conv` + single-step train | ✅ | ✗ |
| CLI: SpeechNet + `--dataset/--bn-frozen/--stratified` | ✅ (modes `infer/train/train_single_step`) | different |
| `shape_optimizer.infer_shapes_with_custom_ops` (+ `all_custom_ops` registry) | ✅ | ✅ (has mezo ops) |
| `zo_transform.py`, `operators/perturb*`, `onnx_node_implementations.py` | ✗ | ✅ |
| `base_exporter.ExportMode.ZO_TRAINING` + `export_zo_training` | ✗ | ✅ |

## 3. Part A — Port the ZO implementation (mostly additive)
**A1. Copy new files verbatim from `ZO` (they don't exist on the base):**
- `onnx4deeploy/transform/zo_transform.py` (perturbation-inject + weight-update + CE-loss)
- `onnx4deeploy/operators/perturb{normal,uniform,triangle,rademacher,eggroll}.py` + `rqsperturb{rademacher,uniform}.py`
- `onnx4deeploy/utils/onnx_node_implementations.py` (`run_onnx_graph` + device RNG `_perturb_rademacher` + node handlers)
- tests: `tests/models/test_zo_perturbation.py`, `tests/operators/test_perturbation_operators.py`, `tests/models/onnx_node_implementations.py`

**A2. Merge (additive — keep the BNFRozen features, add ZO):**
- `core/base_exporter.py`: add `ExportMode.ZO_TRAINING`; add ZO paths in `setup_paths` (`network_zo_train`,
  `network_zo_update`); add the `export_zo_training()` method (which calls **`create_training_test_data_zo()`**
  for fixtures — see B4, NOT the demo's random-data `_create_test_data`); import
  `generate_zo_graph`/`generate_weight_update_graph`. **Preserve** `export_training`,
  `export_training_single_step`, BN-frozen, `create_training_test_data`, `_frozen_pytorch_reference`.
- `Onnx4Deeploy.py` (CLI): add `zo-train`/`q-zo-train` to `-mode` choices, add `--noise-type`, dispatch to
  `export_zo_training(noise_type, quant)`. (SpeechNet is already registered; the data flags already exist.)
- `optimization/shape_optimizer.py`: add the mezo `Perturb*`/`RQSPerturb*` ops to the existing `all_custom_ops`
  registry so `infer_shapes_with_custom_ops` handles the ZO graph.
- `operators/__init__.py`, `utils/__init__.py`, `models/__init__.py`: register the perturb operators +
  `run_onnx_graph`; keep the guarded/lazy quant-exporter imports so non-quant ZO works without DeepQuant
  (the pattern we validated in the `ZO`-branch demo).

## 4. Part B — SpeechNet ZO additions (the real work)
**B1. `zo` config + trainable set (small).** Add a `zo: {epsilon, seed, exceptions}` block to
`SpeechNetExporter.load_config`. `get_trainable_params("full")` already yields **conv (w,b) + BN (γ,β) + fc
(w,b)** = the 22 tensors — no change needed. Confirm `export_zo_training` runs for SpeechNet (inherited).

**B2. ⭐ Perturb BatchNorm γ/β (core change).** The ZO perturb filter in `zo_transform.inject_perturbation_nodes`
is `{Conv, Gemm, MatMul, RequantShift}` — it **skips BatchNorm** (the paper's models are BN-free/GroupNorm).
SpeechNet must **train BN γ/β**, so:
- Add `"BatchNormalization"` to the op-type filter, and perturb **only input idx 1 (scale = γ) and idx 2
  (bias = β)** — **never idx 3 (running_mean) or idx 4 (running_var)** (mirror the `RequantShift`
  `if i != 2: continue` special-case).
- In `generate_weight_update_graph` (perturbs every initializer whose name contains `weight`/`bias`): BN γ is
  `blocks_i_1_weight`, β is `blocks_i_1_bias` → already included; running_mean/var are named `..._running_mean`
  / `..._running_var` → excluded by name. **VERIFY the exact names** (B-open-1) and, to be safe, add
  running_mean/var to `zo_config["exceptions"]`.

**B3. Frozen running stats (keep BN unfolded, eval-mode).** Export the SpeechNet inference ONNX with BN
**unfolded** (`fold_bn=False`, strategy `full`) so a `BatchNormalization` node remains, using the checkpoint's
`running_mean/var` as **frozen initializers**. Because we perturb only γ/β and never mean/var, and
`run_onnx_graph`'s BatchNorm handler already normalises with the initializer mean/var (frozen), the ZO forward
is exactly "train γ/β, freeze stats" — matching BP/exp17/exp18. (Do **not** use `_fold_bn_into_conv`; that
removes BN and its γ/β.)

**B4. Dedicated ZO fixture function `create_training_test_data_zo` (per your guidance — do NOT overload the
demo's random-data `_create_test_data`).** Add a new `create_training_test_data_zo()` that **mirrors the BP
`create_training_test_data`** but for the ZO graphs — same two-tier design (generic in `base_exporter`,
overridable in `SpeechNetExporter` exactly like the BP one at `speechnet_exporter.py:333`):
- **Inputs:** pull **real SilentWear windows + labels** via `get_data_source().load_batches(...)` (same
  S01/fold3/vocalized 54-window seed-42 convention as the BP fixtures), plus the initial weight/bias
  initializers from `network_infer.onnx` (via `_load_init_map`). Save `inputs.npz` = data + `label` + initial
  params (+ any ctrl tensors).
- **Reference outputs:** run **`network_zo_train.onnx`** through the pure-Python **`run_onnx_graph`** (executes
  `Perturb*` + frozen-BN + `SoftmaxCrossEntropyLoss`) → the perturbed-forward **reference loss**; and run
  **`network_zo_update.onnx`** through `run_onnx_graph` → the **updated weight tensors** (the in-place step
  applied to the initial params). Save `outputs.npz` = `loss` + updated params. (Contrast BP: ORT on
  `network_train` + manual SGD; ZO: `run_onnx_graph` on the two ZO graphs, no ORT/autodiff.)
- **Frozen-BN is automatic:** `run_onnx_graph`'s `BatchNormalization` handler normalises with the frozen
  initializer mean/var and we perturb only γ/β — so no BN reference specialisation is needed (unlike the BP
  path's `_frozen_pytorch_reference`). Only add a `SpeechNetExporter.create_training_test_data_zo` override if
  a SpeechNet-specific reference (e.g. balanced-accuracy print, multi-window) is wanted.
- **Wiring:** `export_zo_training()` calls `self.create_training_test_data_zo()` (parallel to how
  `export_training()` calls `self.create_training_test_data()`).

## 5. Part C — Validation (Onnx4Deeploy scope)
1. `python Onnx4Deeploy.py -model SpeechNet -mode zo-train --noise-type rademacher --dataset silentwear
   --bn-frozen-stats --training-strategy full --subject S01 --session 3 --condition vocalized --data-size 54
   --stratified -o <dir>`.
2. Inspect `network_zo_train.onnx`: `Perturb*` on **Conv w/b + BN γ/β + fc w/b** (count matches 22), **no**
   perturb on running_mean/var; `BatchNormalization` present with frozen mean/var; `SoftmaxCrossEntropyLoss`;
   inputs `input`+`label` → scalar loss.
3. Inspect `network_zo_update.onnx`: in-place perturb of the 22 trainable tensors only.
4. Reference loss: `run_onnx_graph` gives a **frozen-BN** perturbed-forward loss; cross-check against a PyTorch
   frozen-BN + Rademacher-perturbation reference (reuse exp18's `rng_mode="device"` perturbation for one seed)
   → should match to fp tol.

## 6. Out of scope (Deeploy / TrainDeeploy, later)
- The C `Perturb*` / weight-update **kernels**, tiling for the new ops, and the **ZO training runner** (the
  shared-`z` window loop, scalar accumulation, `q` directions, ε-sign flip, seed override). Onnx4Deeploy only
  emits the graphs + reference; the runner drives the estimator.

## 7. Open questions to verify during implementation
- **B-open-1:** exact BN initializer names in SpeechNet's exported ONNX (γ/β vs running_mean/var) → fixes the
  perturb-input indices + the `exceptions` list.
- **B-open-2:** does SpeechNet's ONNX emit `BatchNormalization` (idx 1..4 = scale,bias,mean,var) as assumed,
  or a fused/renamed form? Adjust B2 accordingly.
- **B-open-3:** does `base_exporter._create_test_data`'s ZO branch need a SpeechNet override for real data +
  frozen-BN reference, or can the generic path be parameterised?
- **B-open-4:** quantized (`q-zo-train`) path needs DeepQuant + `scales_path`; defer unless needed (we target
  FP32 first, matching exp18).

## 8. Sequencing
A1 (copy) → A2 (merge base_exporter/CLI/shape_optimizer/inits) → smoke-export SpeechNet `zo-train` (expect it
to run but NOT yet perturb BN) → B2/B3 (BN γ/β + frozen stats) → B4 (real data) → C (validate) → then hand the
graphs to the Deeploy side. Each step QW-marked; commit per logical step once reviewed.
