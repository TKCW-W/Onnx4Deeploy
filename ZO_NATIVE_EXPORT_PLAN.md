<!-- 2026-08-04  Plan: make the ZO export emit TrainDeeploy-native training graphs (no downstream prep). -->
# ZO export → BP-native training graph — implementation plan (feat/ZO)

**Root cause:** BP's native structures come from ORT `artifacts.generate_artifacts` (base_exporter.py:844). ZO can't call it (MeZO has no autodiff graph) — it post-processes `network_infer.onnx` via `zo_transform.py`, inheriting the *inference* frontend. So we replicate BP's 4 structures by hand.

Files: `onnx4deeploy/transform/zo_transform.py`, `onnx4deeploy/core/base_exporter.py`, `onnx4deeploy/models/speechnet_exporter.py`, `onnx4deeploy/utils/onnx_node_implementations.py`.

## Step 1 — 22 trainable weights: initializers → graph INPUTS (both graphs)
- Helper `_promote_initializers_to_inputs(graph, names)`: append a `make_tensor_value_info(nm, init.data_type, init.dims)` to `graph.input` for each trainable name; **remove those initializers** (be byte-identical to BP, where trainable params are runner-supplied inputs with no initializer). Keep non-trainable (BN running_mean/var, `_mul` scales) as initializers/Constants.
- **zo_train** (`inject_perturbation_nodes.modify_graph`, ~zo_transform.py:754): collect the perturbed original names (`input_name`, the Perturb node's input) → call helper on `new_graph`. The Perturb consumes the graph-input weight, emits `f"{idx}_{input_name}"` for the forward — same shape as BP (weight input feeds forward op).
- **zo_update** (`generate_weight_update_graph`, ~85, 322-329): currently `inputs=[]`, `outputs=[]`, Perturb out-name==in-name (95). Change: params as **inputs AND outputs**; rename each Perturb output `f"{name}_updated"` and add as graph output (matches BP optimizer-graph in/out contract; also fixes in-place buffer-sharing name-match on device).

## Step 2 — BatchNormalization → BatchNormInternal (5-in/5-out)
At BN re-emission (~zo_transform.py:735-743): emit `op_type="BatchNormInternal"`, `domain="com.microsoft"`, outputs `[Y, Y_run_mean, Y_run_var, Y_saved_mean, Y_saved_inv_std]`, attrs `epsilon`, `momentum`, `training_mode=1`. Add value_info for the 4 extra outputs ([C]). Perturb only γ(1)/β(2); running_mean/var stay frozen (already enforced ~431). Shape handler exists (shape_optimizer.py:836-863). `training_mode=1` matches BP; frozen behavior is the device `BN_FROZEN_STATS` flag (reference uses `model.eval()`).
- Safety: add a `BatchNormInternal` branch to the executor (onnx_node_implementations.py:527) computing frozen-stats BN, returning 5 outputs (only needed for base-stub run_onnx_graph; SpeechNet uses the PyTorch sim).

## Step 3 — canonical 2-output SoftmaxCrossEntropyLoss
Rewrite `append_cross_entropy_loss` (~zo_transform.py:794-849): node `outputs=["loss","log_prob"]` (loss first!), `reduction="mean"`; graph outputs = `loss` `TensorProto.FLOAT []` + `log_prob` `[B,K]`. (run_onnx_graph SCE handler returns log_prob-first — base-stub consumers must fetch by NAME; SpeechNet override unaffected.)

## Step 4 — drop `mezo` domain
Remove `domain="mezo"` on all Perturb nodes (zo_transform.py:101,117,224,266,309,461,476,493,512,553,599). Remove `make_opsetid("mezo",1)` (335,774); opset list → `[("", opset), ("com.microsoft",1)]` (com.microsoft now needed for BatchNormInternal). Perturb ops stay (default domain) — TrainDeeploy already registers `PerturbRademacher` (default domain) from the port.

## Step 5 — pack weight arrays into inputs.npz (SpeechNet `create_training_test_data_zo`, ~speechnet_exporter.py:658-678)
Mirror BP `create_training_test_data`: pack `arr_{idx:04d}` over **all** zo_train graph inputs in graph-input order (`input`, `label`, + 22 weights from `init_map`); `mb{mb}_arr_0000/0001` for other windows (data/label only, `num_data_inputs=2`); keep `meta_*`/`meta_zo_*`. Now `testInitWeights` exist. (Also base-stub base_exporter.py:703.)

## Step 6 — keep PyTorch MeZO reference (no change)
`_zo_pytorch_reference` stays. `outputs.npz` = `{param: updated_weight}` + `log_prob` + `loss_plus` + `loss_minus`. loss_plus/minus are ZO-only (no BP analog). Consider also emitting `loss` = mean(loss_plus,loss_minus) if `generateTrainingNetwork` asserts a `loss` key.

## Verification (before touching TrainDeeploy)
Regenerate the min fixture and assert on `network_zo_train.onnx`/`network_zo_update.onnx`:
- zo_train inputs = `input,label` + 22 weights (24 inputs); 0 trainable initializers.
- 5 `BatchNormInternal` (com.microsoft), 0 plain `BatchNormalization`.
- SCE: 2 outputs, graph outputs `[loss[], log_prob[B,K]]`.
- 0 nodes with domain `mezo`; opset has com.microsoft, no mezo.
- zo_update: 22 inputs + 22 `_updated` outputs.
- inputs.npz has `arr_0000..arr_0023` (24), meta_* + meta_zo_*.
Then run through TrainDeeploy's EXISTING training pipeline (no zo_graph_prep) → single-step n_accum=1 smoke.
