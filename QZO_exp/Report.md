# Report — Quantized ZO (QZO) rework log

Running log for the `/loop` run that reworks QZO into a true extension of the float ZO + shipped quantized
pipelines (offline int8 weights, per-channel RequantShift, weights-as-inputs). See `Plan.md` for the design.

---

## Iteration 1 — orient, lock design, study (2026-08-27)

### Established facts
- Both repos on `feat/QZO`. Containers up: `agitated_hugle` (export), `traindeeploy` (sim).
- Mounts: host `/Users/qiwenwu/ETH` → `agitated_hugle:/app`, `traindeeploy:/app/ETH`.
- **Real data + pretrained weights verified present on host:**
  - checkpoint `…/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt` (223 KB, present).
  - data `…/SilentWear/SilentWear_data/data_raw_and_filt` (S01–S04 present).
- Shipped `q-zo-train` mode exists at `Onnx4Deeploy_ZO/Onnx4Deeploy.py:369` (under study).

### Key technical finding (drives the rework)
Read `onnx4deeploy/transform/zo_transform.py` in full. The **float `inject_perturbation_nodes` already IS the
transform we want**, and its `rqs_rademacher` branch already handles the quantized case per-tensor:
- Conv/Gemm int8 weight → `RQSPerturbRademacher` (div=2¹⁵, n_levels=256) — `zo_transform.py:567-605`.
- RequantShift int32 bias (`add`, input idx 2) → `RQSPerturbRademacher` (div from node, n_levels=2³²) —
  `zo_transform.py:571-579`, gated by `if node.op_type == "RequantShift" and i != 2: continue` (476-477).
- `BatchNormalization → BatchNormInternal` re-emit (com.microsoft, training_mode=1, 5 outputs) — `783-812`.
- perturbed initializers **promoted to graph INPUTS** — `_promote_initializers_to_inputs` (852) + helper (21).
- `append_cross_entropy_loss` shared, label placed at input index 1 — `886-967`.

**Gap identified:** `inject_perturbation_nodes` applies ONE global `noise_type`. Quant SpeechNet needs MIXED
noise — int8 `rqs_rademacher` on Conv/Gemm weights + RequantShift bias, but **float `rademacher`** on the fp32
BatchNormalization γ/β. This per-tensor-dtype noise selection is the core code change for the rework.

### Design decision (locked, per user)
- Quantize weights **offline (standard)**: int8 weight initializers + per-channel RequantShift. Online
  fp32-master perturbation **deferred** as a later optimization (needs a per-channel Quant kernel Deeploy
  lacks).
- **Retire the from-model rebuilder** `build_qzo_int8_graph`; **transform the shipped integer graph** instead
  (true extension). Comment-out, not delete.

### Pipeline map (from study agent, verified)
- **Integerization lives in Onnx4Deeploy**, not TrainDeeploy: `create_quant_pipeline` (12 passes,
  `onnx4deeploy/core/optimization_passes.py:487`) runs inside `export_quantized` (`base_exporter.py:1401`).
  `-mode quant` → `export_quantized()` → `exportBrevitas` (QCDQ) → 12 passes → `network.onnx`. TrainDeeploy
  ingests the finished graph.
- The shipped **q-zo reference (Onnx4Deeploy_ZO, QMCUNet) never ran the 12 passes** — it fed the raw
  `exportBrevitas` QCDQ straight into `generate_zo_graph`. That's the historical gap.
- DeepQuant is NOT pip-installed in `agitated_hugle`; source is on disk. **Fix: `export
  PYTHONPATH=/app/Onnx4Deeploy/DeepQuant` before any quant export.** brevitas 0.13.0 present.

### Ran the real-data quant export (validates path + reveals structure)
```
export PYTHONPATH=/app/Onnx4Deeploy/DeepQuant
python3 Onnx4Deeploy.py -model SpeechNet -mode quant --dataset silentwear \
  --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights <fold_3.pt> --subject S01 --session 3 --condition vocalized --batch 1 \
  -o /app/Onnx4Deeploy/QZO_exp/exp2/qinfer
```
→ `QZO_exp/exp2/qinfer/network.onnx` (12 passes ran, 315→187 nodes). Pretrained + real SilentWear data OK.

### STRUCTURAL REALITY of the integer graph (drives the redesign) — evidence
Inspected `qinfer/network.onnx`: input **int8**, output f32.
- OPS: `Conv×5, Relu×5, MaxPool×3, GlobalAveragePool, Gemm, RequantShift×6, Quant×5, Dequant×5`,
  plus `Div×12 Add×12 Round×12 Clip×12 Sub×12 Mul×12` and `Cast×24`.
- Init dtypes: **12 f32 (the conv/fc weights+biases) + 12 i32 (RQS mul/add)**.
- **Conv weights are STILL f32 initializers behind a QCDQ chain:** `blocks.N.conv.weight (f32) → Div → Add →
  Round → Cast → Cast → Clip → (dequant) Sub → Mul → Conv`. The **Conv consumes the dequant Mul output**, not
  the initializer. The `Cast×2` between Round and Clip **break the constfold pattern**, so the weight QCDQ was
  NOT folded to an int8 initializer.
- **BN is FOLDED** — zero `BatchNormalization` nodes. Trainable params = 5 conv w + 5 conv b + fc w + fc b =
  **12**, no BN γ/β.

### How the SHIPPED q-zo injects perturbation (the target wiring) — evidence
Inspected `Onnx4Deeploy_ZO/QZO_exp/exp1/fixture/network_zo_train.onnx` (QMCUNet, 86 RQSPerturb):
- Weight init feeds the Conv **directly through RQSPerturbRademacher** — `weight_init → RQSPerturbRademacher
  (div=32768, n_levels=256, signed=1) → Conv`. **No weight QCDQ chain** (weights were constfolded to int-valued
  f32 feeding conv directly); only *activations* keep Quant/Dequant/RequantShift.
- One RQSPerturb per conv weight AND bias (bias = int32 RQS). Graph inputs are only `input`+`label` there
  (weights stay initializers in the shipped ref — we instead promote to INPUTS per our float-ZO convention).

### THE GAP (concrete, to close next iteration)
For `inject_perturbation_nodes` (which perturbs initializers found *directly* on Conv/Gemm inputs) to work
like the shipped q-zo, the weight QCDQ must be **constfolded** so the conv weight is an int-valued initializer
feeding the Conv directly. Our SpeechNet export doesn't fold it (the `Cast×2` block the pattern). Options:
- **(A)** Extend the constfold pass (`QuantConstfoldQuantOfInitializerPass`) to tolerate the `Round→Cast→Cast→
  Clip` shape → int8 weight initializer. Then `-mode quant` yields the shipped-style graph and
  `inject_perturbation_nodes` perturbs weight/bias inits directly.
- **(B)** Add a targeted pre-pass in the QZO path that constfolds each weight/bias QCDQ chain to an int-valued
  initializer feeding the consumer directly (mirrors the shipped q-zo form), then perturb.

### Decisions taken / to confirm
- **BN folded (v1):** the quant export folds BN, matching the shipped q-zo (no BN γ/β training). This means
  **no mixed float/RQS noise needed** — all 12 params are int-valued → all `RQSPerturbRademacher`. Earlier
  "mixed noise" plan is moot for v1. BN-unfolded fp32 γ/β training is a **follow-up** (needs keeping BN
  unfolded through export). Flagged, reversible.
- Weights-as-INPUTS still holds (promote via `_promote_initializers_to_inputs`), unlike the shipped ref which
  keeps them initializers.

### Next iteration
1. Choose (A) vs (B) — lean **(B)** (localized to QZO, no risk to the trusted `-mode quant` inference path).
2. Implement the weight-QCDQ constfold + `inject_perturbation_nodes` (rqs_rademacher) + weights-as-inputs +
   SCE loss → `network_zo_train.onnx`; `generate_weight_update_graph` → `network_zo_update.onnx`.
3. Wire `_export_qzo_training` to this (comment out `build_qzo_int8_graph`). Export real-data fixture to
   `QZO_exp/exp2/`. Host reference via `run_onnx_graph`.
4. Re-examine whether BN can be kept unfolded (follow-up scope).

### Housekeeping
- Temp inspection scripts written under the mount root (`/Users/qiwenwu/ETH/_qzo_*.py`) — removed at end of
  iteration.
- `agitated_hugle` quant export requires `PYTHONPATH=/app/Onnx4Deeploy/DeepQuant` (recorded).

---

## Iteration 2 — root-cause the weight integerization; scope the fix (2026-08-27)

### ROOT CAUSE of the un-integerized weights (definitive, code-level)
- `create_quant_pipeline` pass #3 `fold_qcdq_to_quant_dequant` (`optimization/qcdq_to_deeploy.py:148`) collapses
  `Div→Add→Round→Clip → Quant` **only for per-TENSOR scales**: its `_const_scalar` returns `None` when
  `v.size != 1` (`:165-167`), so a **per-channel weight scale (size C) is skipped** and the weight QCDQ stays
  raw. Pass #4 `constfold_quant_of_initializer` (`:502`) only folds a *Quant node* whose input is an
  initializer — since the weight never became a Quant, it's never integerized. Verified: `network.onnx`
  activations fold to `Quant×5` but conv/fc weights remain f32 `Div→…→Mul→Conv` (0 int8 initializers).
- The Casts in the weight chain are only on the **Clip bounds** (`Cast(Constant)`), not the data path — so the
  sole blocker is the **per-channel scale**, exactly the per-tensor limitation from the earlier discussion.
- The requant builder `fold_dequant_quant_to_requantshift` (`:290`) computes `mul` from **activation scales
  only** (`scale_d/scale_q`, scalar, `:330`). There is **no path for the weight scale `s_w` to enter the
  post-conv RequantShift**, and for per-channel `s_w[c]` that RequantShift `mul` would have to be per-channel.
  ⇒ The existing pipeline integerizes **per-tensor** weights end-to-end but silently leaves **per-channel**
  weights as fp32 QCDQ (device would run fp32 conv — the QZO_mixed Bug A/B).

### Shipped QMCUNet q-zo — how it actually represents weights (ground truth)
Inspected `Onnx4Deeploy_ZO/QMCUNet-Rad/network_infer.onnx` (the proven q-infer base) around the first Conv:
- Activations are fully QCDQ'd (`Div/Add/Round/Clip → Sub/Mul`, per-tensor scale 1/128).
- **The conv weight is a RAW FLOAT initializer** (`[16,3,3,3]`, ±0.07, `intval=False`) fed **directly** to
  Conv — **no weight QCDQ chain at all**. In `network_zo_train.onnx` this float weight goes through
  `RQSPerturbRademacher` (per-channel `mul[16]`) straight into Conv.
- Implication: the shipped q-zo does **not** integerize weights in the ONNX; it ships **float weights +
  activation QCDQ** and the per-channel weight scale is carried in the RQSPerturb `mul` (and presumably applied
  by Deeploy at lowering). Our SpeechNet `exportBrevitas` instead emits **weight QCDQ chains** — a real
  export-form divergence to reconcile.

### Open design question (drives next step)
Where does per-channel weight quant physically happen for a true device int8 conv?
- **(P-chan proper)** Extend the pipeline: fold per-channel weight QCDQ → int8 initializer AND emit a
  **per-channel post-conv RequantShift `mul[c] = s_in·s_w[c]/s_out`** (RequantShift already supports
  per-channel mul — shipped). This is the correct, supervisor-aligned design; needs 2–3 pass edits + on-device
  validation.
- **(Per-tensor v1)** Use per-tensor weight quant → the existing pipeline integerizes end-to-end **today**
  (int8 weight init + scalar RequantShift, Deeploy merges to int8 conv). Fast, conservative, validated; loses
  a little accuracy. Per-channel becomes a clean follow-up.
- **(Shipped-mirror)** Reproduce QMCUNet's float-weight + activation-QCDQ form and let Deeploy quantize the
  weight — needs understanding the RQSPerturb/Deeploy weight-quant semantics (not yet nailed).

**Chosen direction:** pursue **P-chan proper** (matches the user's explicit per-channel requirement and the
supervisor). The unambiguous first building block — needed by P-chan — is a **per-channel weight/bias
constfold**: evaluate each weight/bias `Div→Add→Round→Clip` on the constant → int8/int32 initializer feeding
the Conv/Gemm directly, and return the per-channel `s_w[c]` map for the RequantShift absorption. Implement +
unit-test that first (this iteration), then wire the per-channel RequantShift and validate.

### Iteration-2 OUTCOME — per-channel weight/bias constfold IMPLEMENTED + VERIFIED
New module **`onnx4deeploy/transform/qzo_weight_integerize.py`** —
`integerize_perchannel_weights(model) -> (model, scale_map)`:
- Traces each Conv/Gemm weight(input[1]) & bias(input[2]) back through
  `Mul<-Sub<-Clip<-Round<-Add<-Div<-init` (scales/zp/bounds resolved whether they're initializers OR
  Constant/Cast outputs — the disambiguation that took two fixes: bounds via `Cast(Constant)`, and Div source =
  the initializer input vs the Constant scale).
- Evaluates per-channel int8 weights / int32 biases, rewires the consumer to read the int initializer
  **directly**, deletes the dead QCDQ+dequant nodes, returns `scale_map[conv] = {weight_scale s_w[C],
  bias_scale s_b[C], weight_src, bias_src}`.

**Smoke test on the real `qinfer/network.onnx` (verified):** 6 weights → **int8 inits consumed directly** by
Conv/Gemm; biases → **int32**; **all 12 weight/bias QCDQ chains removed** (residual `Div/Add/Round/Clip/Sub/Mul
= 0`); activation `Quant×5/Dequant×5/RequantShift×6` preserved; per-channel `s_w` (size = out-channels
8/16/16/32/32/9) + `s_b` returned. INIT dtypes now `i8×6, i32×18, f32×12(untouched activation consts)`.

Committed as WIP (module not yet wired into the export; the RequantShift `s_w` absorption + numerical
validation are the next step, below).

### Next iteration (3)
1. **Per-channel RequantShift `s_w` absorption.** Removing the weight dequant dropped the per-channel `s_w[c]`
   factor from the conv output; the post-conv activation RequantShift `mul` must absorb it →
   `mul[c] = s_in·s_w[c]/s_out` (per-channel vector; RequantShift supports it). Build/patch this from
   `scale_map`, then **numerically validate**: integerized int8-conv graph output == original QCDQ graph output
   (ORT, same input) within quant tolerance. This is the delicate correctness gate.
2. Once numerics match: `inject_perturbation_nodes(rqs_rademacher)` on the int8 weights + `_promote_
   initializers_to_inputs` (weights-as-INPUTS) + `append_cross_entropy_loss` → `network_zo_train.onnx`;
   `generate_weight_update_graph` → `network_zo_update.onnx`; wire `_export_qzo_training` (comment out
   `build_qzo_int8_graph`); export real-data fixture to `QZO_exp/exp2/`.
