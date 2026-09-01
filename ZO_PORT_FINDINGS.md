<!-- SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna ; SPDX-License-Identifier: MIT -->
# ZO port findings — SpeechNet zeroth-order (MeZO) on `feat/ZO`

**Date:** 2026-08-04
**Branch:** `feat/ZO` (from `feat/BNFRozen_OptionB`; ZO code ported from the `ZO` branch @ `147285e`)
**Scope:** the **Onnx4Deeploy** side only — generate functionally-correct SpeechNet ZO graphs + a
device-faithful reference fixture. The Deeploy/TrainDeeploy kernels + ZO **runner** are out of scope (see §8).

---

## 1. What the ZO export produces
`python Onnx4Deeploy.py -model SpeechNet -mode zo-train --noise-type rademacher --bn-frozen-stats
 --dataset silentwear --pretrained-weights <ckpt> --subject S01 --session 3 --condition vocalized
 --data-size 54 --stratified --lr 3e-6 --n-accum 4 --n-epochs 40 -o <dir>` →

| file | what it is |
|---|---|
| `network_infer.onnx` | plain forward (BN **unfolded + frozen stats**, γ/β as separate initializers) |
| `network_zo_train.onnx` | **single-step template**: each trainable weight → `PerturbRademacher` → forward → `SoftmaxCrossEntropyLoss` → `log_prob`. Inputs `input`,`label`. |
| `network_zo_update.onnx` | **single-step template**: one in-place `PerturbRademacher` per trainable weight (`out-name == in-name`). No I/O. |
| `inputs.npz` / `outputs.npz` | multi-step reference fixture (§5) |

Perturbed set = **22 tensors**: Conv w/b (10) + **BN γ/β (10)** + fc w/b (2); running mean/var **never** perturbed.

## 2. ⭐ Key architectural fact: graphs are single-step templates; the runner owns the loop
The two ZO graphs encode **one perturbed forward + loss** and **one in-place perturbation**. Everything about
the *training loop* is the **runner's** responsibility (not in the ONNX):

- **±ε passes.** `network_zo_train` bakes `eps=+0.01` and the executor uses `sign=+1` — it is the **+ε forward
  only**. The −ε pass is the **same graph run again with the sign flipped** (device Perturb kernel takes a
  `dir` flag, paper Alg. 2). What links +ε and −ε is the **same seed** → same `z`. `g_proj=(L₊−L₋)/2ε`.
- **Gradient accumulation (n_accum).** There is **no per-weight accumulation buffer** (unlike BP's
  `InPlaceAccumulatorV2`). ZO accumulates a **single scalar**: because all n_accum micro-batches share one `z`
  (same seed), `ĝ = [Σᵢ(L₊ⁱ−L₋ⁱ)/(2εN)]·z` = one scalar × one shared direction. The runner sums the scalar
  loss differences (O(1) float), then runs `network_zo_update` **once** with coefficient `−lr·g_proj`. This is
  ZO's memory win: O(1) vs BP's O(#params)×n_accum.
- **Seed override / q / lr.** Per-step seed (convention `g = seed + step·q + q_i`), the number of query
  directions `q`, and the learning rate all live in the runner; the graph only carries defaults.

So `PerturbRademacher` attrs are just `{idx (node_id), seed, eps}`; the runner supplies sign, per-step seed,
and the update coefficient.

## 3. Bugs found & fixed
### 3a. ⚠️ Perturb `node_id` (idx) inconsistent between zo_train and zo_update — **pre-existing, critical**
`_exec_mezo` seeds each param's `z` by `scramble(seed + 8·node_id + core)` with `node_id = attrs["idx"]`. For
MeZO to work, the ±ε probe (`zo_train`) and the update (`zo_update`) **must use the same `z`** per param — so
their `idx` must match. But the original code assigned them **independently**: `inject_perturbation_nodes`
bumps its counter **by 2** per weight, `generate_weight_update_graph` **by 1** → e.g. `fc_bias` = idx 42 in
train vs 21 in update. **21/22 params mismatched** (SpeechNet); **9/10** in the LightweightCNN ZO-branch demo
too — so it is **pre-existing in the ZO branch**, not introduced by the port. On device this makes the update
step along an *unrelated* random direction for ~90% of tensors → MeZO ≈ random walk.
**Why never caught:** the ZO branch only ran single-graph / single-perturbation tests, never the full
forward+update loop; exp18 used its own consistent RNG. Our multi-step fixture is the first thing that requires
`z_forward == z_update`.
**Fix (`c1826f7`):** key `idx` on a **canonical per-parameter id** = the tensor's position in
`network_infer`'s initializer list. Both functions read `network_infer`, so the same param gets the same
`node_id` in both graphs → **0/22 mismatch**. Shared-code fix (corrects all models + the demo).

### 3b. `run_onnx_graph` BatchNormalization handler broke on NCHW
The ported executor never handled BN (the paper's models are BN-free / GroupNorm). Its BN handler multiplied a
`(C,)` scale against `(N,C,H,W)` → broadcast error. **Fix:** reshape the per-channel params to `(1,C,1,…)`.
Frozen-stats eval BN then computes bit-exactly (validated vs ORT, 1.5e-8).

## 4. SpeechNet-specific additions (the "train BN γ/β, freeze stats" requirement)
- **BN must stay UNFOLDED.** `torch.onnx.export` in eval + constant-folding folds eval-BN into Conv, deleting
  γ/β. `export_zo_training` instead exports in **training mode with BN set to eval** (`bn_frozen_stats` →
  `TrainingMode.PRESERVE`, `do_constant_folding=False`), mirroring `export_training`. Result: 5 standalone
  `BatchNormalization` nodes, γ=`blocks_i_1_weight`, β=`blocks_i_1_bias`, frozen `running_mean/var`.
- **Perturb BN γ/β only.** `inject_perturbation_nodes`'s op filter gains `"BatchNormalization"`, perturbing
  **only scale (idx 1) and bias (idx 2)** — never running_mean (3) / running_var (4). So BN affine trains,
  stats stay frozen. (`generate_weight_update_graph` already caught BN γ/β by the `weight`/`bias` name rule.)
- **`zo` config block** in `SpeechNetExporter.load_config`: `{epsilon: 0.01, seed: 42, exceptions: []}`.

## 5. Multi-step reference fixture (`create_training_test_data_zo`)
Mirrors the **BP `create_training_test_data`** flow (not the demo's single-perturbation `_create_test_data`):
`SpeechNetExporter.create_training_test_data_zo` + `_zo_pytorch_reference` simulate the **full N-step faithful
MeZO** in PyTorch (frozen-BN, shared-`z` per window via the **device RNG** `_perturb_rademacher`, scalar
`(L₊−L₋)` accumulation over n_accum, `θ ← θ − lr·g_proj·z` per window, q-averaged; per-window seed
`g = seed + step·q + q_i`).
- **`outputs.npz`** = final trained weights + per-step `log_prob` (device log-softmax output) + scalar
  `loss_plus`/`loss_minus`.
- **`inputs.npz`** = mb0 feed (`arr_`) + every other window/label (`mb{mb}_arr_`) + `meta_*`
  (data_size, n_batches, n_accum) + ZO meta (`eps, seed, lr, q`).
- CLI threads `--n-batches/--n-steps/--n-epochs/--n-accum` into `zo-train`.

## 6. Validation (real S01/fold3 checkpoint + real SilentWear data)
- Forward bit-exact vs ORT incl. frozen BN (**1.5e-8**).
- 22 perturb targets incl. BN γ/β, never mean/var; **zo_train ≡ zo_update perturb sets** (0/22 idx mismatch).
- `network_zo_update`: `updated − original = ±ε` exactly (Rademacher applied to all 22 tensors).
- Reference **step-0** perturbed forward matches `run_onnx_graph(network_zo_train)` **bit-exactly (1e-6)** →
  the PyTorch reference is **device-faithful**.
- Persistent fixture at `onnx/model/speechnet_zo_S01_fold3/` (lr 3e-6, ε 0.01, n_accum 4, **40 epochs / 540
  steps**): loss trends **0.665 → 0.500**, final weights move `max|Δ|=0.0044` (sensible for lr 3e-6). NOTE: 40
  epochs is the *BP* recipe length; ZO's accuracy-optimal is **200 epochs** (exp18) — regenerate at
  `--n-epochs 200` when a fully-trained reference is needed.

## 7. Commits (on `feat/ZO`)
- `c4ed3c0` — port ZO + SpeechNet BN γ/β ZO (Part A/B, graphs correct).
- `c1826f7` — idx-consistency fix (§3a).
- `db51551` — multi-step fixture + CLI threading (§5).
(`network_infer/zo_train/zo_update.onnx` correct; `onnx.checker` passes on `zo_train`; `zo_update`'s in-place
`out==in` SSA "violation" is by design.)

## 8. Deferred — the TrainDeeploy ZO runner (the actual training loop)
Onnx4Deeploy only emits graphs + reference. The runner (out of scope here) must implement, per §2:
1. run `network_zo_train` **twice per micro-batch** (`+ε`, `−ε`), **same per-step seed**;
2. **scalar-accumulate** `(L₊−L₋)` over n_accum (one float), average over q;
3. run `network_zo_update` **once** with coefficient `−lr·g_proj` (same seed → same `z`);
4. per-step seed convention `g = seed + step·q + q_i` (must match the reference fixture's convention).
Also needs the C `Perturb*` / weight-update kernels + tiling for the mezo ops.
**Spec anchors (from exp18):** shared-`z` scalar accumulation, **q=1 sufficient**, **lr ≈ 3e-6**, ε ≈ 0.01.

## 9. Open items / caveats
- Fixture uses `config.learning_rate` (pass `--lr 3e-6`); default is BP's 0.001.
- `onnx/` is gitignored → the persistent fixture is on disk but **untracked** (move to a `Tests/`-style path to
  commit as a canonical reference).
- Base `create_training_test_data_zo` (non-SpeechNet models) remains the simpler single-perturbation version.
- Quantized `q-zo-train` path not wired for SpeechNet (needs DeepQuant + scales); FP32 first.
- `operators/__init__.py` doesn't register the `Perturb*` operator *tests* (only needed for `-operator`, not
  model export).

## References
exp18 (SilentWear `feat/on-device-ZO/exp18_zo_faithful_sim`): faithful MeZO PyTorch sim → ZO 87.36 ≈ BP 86.11.
`demo_bp_vs_zo/TUTORIAL_BP_vs_ZO.md`, `demo_bp_vs_zo/zo_rademacher/ZO_PAPER_STUDY.md` (on the `ZO` branch).
