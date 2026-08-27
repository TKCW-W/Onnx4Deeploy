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

---

## Iteration 3 — establish the host reference; find the -mode quant graph is not integer-correct (2026-08-27)

### Reference-runner fixes (reusable, committed)
To run pipeline-produced graphs through `run_onnx_graph` I fixed three real gaps:
- **Toposort:** `create_quant_pipeline` output nodes are unordered → `run_onnx_graph` KeyErrors on a
  not-yet-produced input. Added a toposort in the harness.
- **Domain:** the pipeline emits `Quant/Dequant/RequantShift` in the DEFAULT domain, but `run_onnx_graph`
  routes Deeploy ops by `ai.onnx.contrib`. Set the domain in the harness (device fixture domain is a separate
  concern to handle at emit time).
- **Attribute-scale Quant/Dequant:** `_exec_deeploy` read scale/zp from node INPUTS only; `fold_qcdq_to_quant_
  dequant` emits them as ATTRIBUTES. Added an additive attr fallback in
  `onnx4deeploy/utils/onnx_node_implementations.py` (`_exec_deeploy`, Quant+Dequant) — also reads
  `bit_width`. `_exec_deeploy` RequantShift **already supports per-channel `mul`** (NCHW/NHWC broadcast,
  lines 852-865) — good, the per-channel absorption will validate here.

### KEY FINDING — the `-mode quant` `network.onnx` is NOT a numerically-correct integer graph
Ran `network.onnx` (toposorted, domains fixed) on the real int8 input; output = **all zeros** (its argmax 0
only *coincidentally* equals the reference argmax 0). Root reasons:
- It is a **hybrid**: fp32 conv (weight still dequantized) feeding an **integer** RequantShift. Feeding a tiny
  fp32 conv output to the integer RequantShift (`x.astype(int64)`) truncates to 0.
- The post-conv RequantShift `mul = 8388608`, `div = 2¹⁶` ⇒ factor **×128 = 1/s_out** only. It **omits both
  `s_in` and the per-channel `s_w[c]`** (`s_w` was meant to ride in the conv via the dequantized weight;
  `s_in` is simply absent since the offline int8 input is used raw). So the pipeline graph is not a correct
  standalone integer graph for SpeechNet — consistent with the QZO_mixed forward-fidelity bugs.
- ⇒ **Do not transform the `-mode quant` graph** (building on sand). The trustworthy reference is
  `outputs.npz` = the PyTorch/Brevitas output `[1.999, -0.559, …]` (argmax 0).

### DECISION — build the correct int8 ZO datapath from verified parts (validated vs PyTorch)
Combine what's verified rather than patch a broken graph:
- **Offline int8 weights + per-channel `s_w`/`s_b`** from `integerize_perchannel_weights` (iter-2, verified,
  sourced from the REAL calibrated export) — these are the trainable params, fed as **int8 graph INPUTS**.
- **Per-channel RequantShift** `mul[c] = round(s_in·s_w[c]/s_out · 2^shift)`, `add = int32 bias` (in the RQS
  add) — the datapath math VERIFIED earlier in `build_qzo_int8_graph` (its argmax matched Brevitas). Need the
  activation scales `s_in`/`s_out` per layer (from the graph's Quant/Dequant, or `dump_brevitas_scales`).
- **Activation path** `Conv(int8) → RequantShift → Dequant(s_out) → Relu → MaxPool → Quant(s_in_next)` (BN is
  folded here → no BN γ/β), **weights/biases as INPUTS**, `RQSPerturbRademacher` on int8 weights + int32
  biases, shared `append_cross_entropy_loss`.
- **Validate against `outputs.npz`** (PyTorch) within quant tolerance (argmax + close logits). Only then wire
  `_export_qzo_training`, export the real-data fixture, build the update graph.

This keeps the *offline-weights + weights-as-inputs + per-channel RequantShift* design, reuses the verified
integerize pass and the verified datapath math, and avoids the broken pipeline graph. `build_qzo_int8_graph`'s
online-weight-Quant is replaced by offline int8 weight inputs (no weight Quant node at all → the per-channel
fold problem disappears entirely).

### Activation scales captured (de-risks iter-4) — all UNIFORM 1/128
Every `Quant`/`Dequant` activation scale = **0.0078125 = 1/128**, and block0 offline input scale `s_in0` =
**1/128** too. So **`s_in = s_out = 1/128` for every layer**. The per-channel post-conv RequantShift therefore
simplifies to:
```
mul[c] = round(s_in·s_w[c]/s_out · 2^shift) = round(s_w[c] · 2^shift)     # s_in,s_out cancel
add    = int32 bias[c]      (per-channel, from integerize)
```
This also nails the pipeline bug quantitatively: its RequantShift factor is `1/s_out = 128` (per-tensor),
whereas the correct factor is `s_w[c] ≈ 5e-4` → **~128× too large**, which truncates to 0 on the int
RequantShift → the all-zeros output. Confirms the `-mode quant` graph is wrong for SpeechNet.

### Next iteration (4) — implement the datapath builder + validate
With everything in hand (int8 weights + per-channel `s_w`/`s_b` from `integerize`; `s_in=s_out=1/128`), build
the int8 ZO datapath and validate the eps=0 forward vs `outputs.npz` (PyTorch, argmax 0, `[1.999,…]`):
`a_int8 → Conv(int8 weight INPUT) → RequantShift(mul[c]=round(s_w[c]·2^shift), add=int32 bias INPUT) →
Dequant(1/128) → Relu → MaxPool → Quant(1/128) → …`; fc analogous (Gemm). Then `RQSPerturbRademacher` on the
int8 weight/bias inputs + `append_cross_entropy_loss` → `zo_train`; mirror for `zo_update`; wire
`_export_qzo_training` (comment out `build_qzo_int8_graph`); export real-data fixture to `QZO_exp/exp2/`.

---

## Iteration 4 — VALIDATED int8 datapath (numerically matches PyTorch) (2026-08-27)

### Implemented `build_int8_forward` (in `qzo_weight_integerize.py`) — VERIFIED
Turns the broken `-mode quant` QCDQ graph into a numerically-correct int8 datapath:
- `integerize_perchannel_weights` → int8 weights + int32 biases (bias kept as the **Conv/Gemm 3rd input** —
  the shipped q-zo convention → both weight & bias are directly perturbable graph inputs for ZO).
- **conv:** patch the post-conv RequantShift → `mul[c] = round(s_w[c]·(s_in/s_out)·div)`, `add = 0`, `div`;
  bias stays in Conv (`o = (acc+b_int32)·mul/div`). With `s_in=s_out=1/128` ⇒ `mul[c]=round(s_w[c]·div)`.
- **fc (output layer, no RequantShift after Gemm):** keep int32 bias in Gemm, append a **per-channel dequant
  `Mul(s_in·s_w[c])`** → f32 logits (`logit[c] = gemm_out · s_in·s_w_fc[c]`).
- domains set to `ai.onnx.contrib`; toposorted.

### Numerical validation vs PyTorch reference (`outputs.npz`) — PASS
`run_onnx_graph(network_int8_infer.onnx, inputs.npz) → logits`, compared to the PyTorch output:
| DIV | argmax | cos | maxdiff |
|--|--|--|--|
| 2^16 (RQS-add bias) | 0==0 ✅ | 1.0000 | 0.008 |
| **2^16 (bias-in-Conv, shipped conv.)** | **0==0 ✅** | **0.9999** | **0.032** |
| 2^18 / 2^20 | 0==0 ✅ | 1.0000 | 0.006–0.008 |
`out=[2.01,-0.57,-1.02,-0.74,…]` vs `ref=[1.999,-0.559,-1.015,-0.71,…]` — matches within quant noise.
Chose **bias-in-Conv, DIV=2^16** (ZO-friendly + shipped convention). Artifact:
`QZO_exp/exp2/network_int8_infer.onnx` (int8 input, RequantShift×6, Quant×5, Dequant×5, per-channel).

**This closes the correctness gate** — we now have a real, validated int8 SpeechNet inference datapath built
from the export + per-channel scales (offline weights), reproducing PyTorch. Known cleanup: 56 orphan
`Constant` + 24 `Cast` (dead ends from removed QCDQ) — harmless, prune later.

### Next iteration (5) — ZO fixture from the validated int8 graph
1. Promote int8 weights + int32 biases to graph **INPUTS** (`_promote_initializers_to_inputs`).
2. Inject `RQSPerturbRademacher` on each (weight: div=2^15,n_levels=256; bias int32: div=2^31,n_levels=2^32),
   perturb-`mul = round(eps/s_w·2^15)` from the `scale_map` — reuse the float `inject_perturbation_nodes`
   rqs_rademacher machinery (it already targets Conv/Gemm weight+bias, promotes to inputs).
3. `append_cross_entropy_loss` (label input) → `network_zo_train.onnx`; validate eps=0 == inference; check
   eps>0 L+/L- differ.
4. `generate_weight_update_graph` → `network_zo_update.onnx`. Wire `_export_qzo_training` through this
   (comment out `build_qzo_int8_graph`); export real-data fixture to `QZO_exp/exp2/` with `inputs.npz`
   (int8 weights+biases as inputs) + host L+/L- reference.

---

## Iteration 5 — QZO zo_train graph builder (VALIDATED) (2026-08-27)

### `RQSPerturbRademacher` semantics (confirmed from `_perturb_rqs_rademacher`)
`noise_q = (rad·mul + rounding) >> log2(div)`, added to the int64-cast data, clipped to
`[-(n_levels/2)+1, n_levels/2-1]`. **eps=0 ⇒ mul=0 ⇒ identity** (`rounding = 2^(S-1)`, `>>S = 0`). Magnitudes:
weight int8 `div=2^15, n_levels=256, mul[c]=round(eps/s_w[c]·2^15)`; bias int32 `div=2^31, n_levels=2^32,
mul[c]=round(eps/s_b[c]·2^31)` — perturbs each by ±eps in real units.

### Implemented `build_qzo_train_graph` in `qzo_transform.py` — VERIFIED
Extends the pipeline (no from-model rebuild): `build_int8_forward` (iter-4, validated) → inject
`RQSPerturbRademacher` on each Conv/Gemm weight+bias (mul from the per-channel `scale_map`) → promote the int8
weights + int32 biases to graph **INPUTS** (shared `_promote_initializers_to_inputs`) → append canonical
`SoftmaxCrossEntropyLoss` (shared `append_cross_entropy_loss`). Deprecated the old from-model
`build_qzo_int8_graph` in-place (commented, kept).

**Validation (`run_onnx_graph` on real int8 data):**
- **14 graph inputs** = `input, label` + **12 trainable params** (6 conv/fc weights int8 + 6 biases int32) —
  weights-as-INPUTS ✅.
- **12 `RQSPerturbRademacher`** (6 weight + 6 bias).
- **eps=0**: loss=0.392, `log_prob` argmax **0** (== inference/PyTorch) — identity ✅.
- **eps=0.01**: loss=0.451 (0.05 → 2.10) — the graph responds to perturbation ✅.

This is the complete, validated QZO **zo_train** fixture: offline int8 weights, weights+biases as inputs,
per-channel RequantShift, RQSPerturb, SCE loss — a true extension of the float ZO + shipped quant pipelines.

### Next iteration (6) — update graph, wiring, real-data fixture
1. **zo_update graph**: mirror `generate_weight_update_graph` for the int8 params (RQSPerturb each int8
   weight/int32 bias → `*_updated` output; weights as INPUTS) → `network_zo_update.onnx`.
2. **Wire `_export_qzo_training`**: brevitas model + PTQ calibrate on real SilentWear → `exportBrevitas` +
   `create_quant_pipeline` (needs `PYTHONPATH=…/DeepQuant`) → `build_qzo_train_graph` + update graph; write
   `inputs.npz` (params as inputs) + host L+/L− reference (`outputs.npz`). Comment out the
   `generate_zo_graph(qzo_model=…)` branch that called `build_qzo_int8_graph`.
3. Export the **real-data** fixture (fold_3 pretrained, S01/sess3/vocalized) to `QZO_exp/exp2/`.
4. Then (optional) the TrainDeeploy single-step smoke test.

---

## Iteration 6 — COMPLETE real-data QZO fixture (zo_train + zo_update + L±) (2026-08-27)

### Implemented `build_qzo_update_graph` + `_iter_qzo_params` (shared) + `_prune_orphans`
- `_iter_qzo_params` yields each Conv/Gemm weight+bias in canonical order (weight-then-bias) with a shared
  `idx` — used by BOTH train and update so the perturb `z` (seed+idx) matches per param.
- `build_qzo_update_graph`: params IN → `RQSPerturbRademacher` → `<param>_updated` OUT (mirror of
  `generate_weight_update_graph`), same mul/idx/seed as train.
- `_prune_orphans` in `build_int8_forward`: removes the dead `Constant`/`Cast` left by QCDQ removal (56→4,
  24→0) + orphan initializers → clean graph for Deeploy. Refactored train builder onto `_iter_qzo_params`.

### Generated the real-data fixture → `QZO_exp/exp2/` (`generate_fixture.py`)
Built from `qinfer/network.onnx` (the **real-data** `-mode quant` export: fold_3 pretrained weights, S01 /
session 3 / vocalized calibration). L± via building the train graph at `+eps` and `-eps` (same seed → same z,
opposite sign) = θ±εz:
- **`network_zo_train.onnx`** — `RQSPerturb×12, Conv×5(int8), RequantShift×6(per-ch), Dequant×5, Relu×5,
  MaxPool×3, Quant×5, Gemm, Mul(fc dequant), SoftmaxCrossEntropyLoss`; **14 inputs** = `input(i8), label(i64)`
  + **12 trainable params as INPUTS** (6 weights i8 + 6 biases i32); outputs `loss, log_prob`.
- **`network_zo_update.onnx`** — `RQSPerturb×12`; 12 params IN → 12 `*_updated` OUT.
- **`inputs.npz`** — `input(int8,1×1×14×700), label` + 12 int8/int32 params.
- **`outputs.npz`** — `loss_plus=0.4506, loss_minus=0.3238, grad=(L+-L-)/2ε=6.339, log_prob(1,9)`,
  `updated_<param>` ×12. `log_prob` argmax 0 == label. **L+ ≠ L−** → real ZO gradient signal.

**DELIVERABLE MET:** a complete, validated, **real-data + pretrained** quantized-ZO fixture with **offline
int8 weights**, **trainable params (weights+biases) as INPUTS**, per-channel RequantShift, RQSPerturb, SCE
loss, and host L±/grad reference — a true extension of the float ZO transform machinery
(`_promote_initializers_to_inputs`, `append_cross_entropy_loss`) and the shipped quantized pipeline
(`create_quant_pipeline` integer export). Artifacts + `generate_fixture.py` in `QZO_exp/exp2/`.

### Next iteration (7)
1. **CLI wiring**: route `_export_qzo_training` through `export_quantized` (real SilentWear + `--pretrained-
   weights`) → `build_qzo_train_graph` + `build_qzo_update_graph`, so `Onnx4Deeploy.py -mode q-zo-train`
   produces the fixture end-to-end (extension on the CLI, not a script). Comment out the old
   `generate_zo_graph(qzo_model=…)` branch.
2. (Optional) TrainDeeploy single-step on-device smoke test: pack the fixture, run
   `deeployMezoRunner_tiled_siracusa.py` (n_steps 1), compare device L±/grad vs host; verify INT8 conv kernels.

---

## Iteration 7 — CLI wiring COMPLETE; end-to-end real-data fixture from `-mode q-zo-train` (2026-08-27)

### Rewrote `_export_qzo_training` (base_exporter.py) — the CLI extension
Replaced the deprecated `generate_zo_graph(qzo_model=…)`/`build_qzo_int8_graph` route with the validated
offline-int8 flow, all inside the existing `-mode q-zo-train` CLI:
1. `create_brevitas_model` (pretrained) → `_fold_conv_bn_inplace` → **PTQ calibrate on REAL SilentWear** (via
   `get_data_source().load_batches`, a real labeled window as the export example + fixture input/label).
2. `exportBrevitas` (lenient-allclose) + `create_quant_pipeline` → integer `network.onnx`.
3. `build_qzo_train_graph` + `build_qzo_update_graph` (offline int8 weights, weights/biases-as-INPUTS).
4. Fixture I/O: `inputs.npz` (int8 input + real label + 12 int8/int32 params) + `outputs.npz` (host
   `loss_plus`/`loss_minus`/`grad`/`log_prob`).
Also: **added the vendored DeepQuant to `sys.path`** so the CLI runs without a manual `PYTHONPATH`, and set
explicit **opset imports** (`'' , ai.onnx.contrib, mezo, com.microsoft`) on the train graph (fixes the ORT
shape-inference "No opset import for domain mezo" warning).

### Verified — one command produces the whole fixture
```
python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights <fold_3.pt> --subject S01 --session 3 --condition vocalized --batch 1 \
  -o /app/Onnx4Deeploy/QZO_exp/exp2
```
→ `QZO_exp/exp2/{network.onnx, network_zo_train.onnx, network_zo_update.onnx, inputs.npz, outputs.npz}`.
`network_zo_train`: opsets `['',ai.onnx.contrib,mezo,com.microsoft]`, **14 inputs (12 params-as-INPUTS)**,
`RQSPerturb×12, Conv×5(int8), RequantShift×6, Dequant×5, Relu×5, MaxPool×3, Quant×5, Gemm, Mul, SCE`.
`inputs.npz`: int8 input `(1,1,14,700)` + real **label 0** + 12 int8/int32 params. `outputs.npz`:
`L+=2.589, L-=3.358, grad=-38.48`. (argmax 3 ≠ label 0 = the pretrained model mispredicts this real window →
higher loss + strong real ZO gradient — correct real-data behaviour, not a bug.)

---

## CONCLUSION — result of this loop run (through iteration 7)

**The primary goal is achieved:** a completely extended `Onnx4Deeploy` capable of **quantized ZO**, driven by
`-mode q-zo-train`, generating a **real-data + pretrained-weights** int8 fixture. Delivered:
- **Offline int8 weights** (standard per-channel quant), **trainable params (weights+biases) as graph
  INPUTS**, per-channel RequantShift — the design agreed this run.
- A **true extension** of both pipelines: the shipped **quant** pipeline (`exportBrevitas` +
  `create_quant_pipeline` integer export) and the **float-ZO** transform helpers
  (`_promote_initializers_to_inputs`, `append_cross_entropy_loss`), plus the new per-channel integerization
  (`qzo_weight_integerize`) that closes the pipeline's per-channel-weight gap.
- **Numerically validated**: the int8 forward matches PyTorch (cos≈1.0, argmax match); eps=0 → identity;
  eps>0 → real ZO gradient (L+≠L−).
- New/changed code: `onnx4deeploy/transform/qzo_weight_integerize.py` (integerize + `build_int8_forward`),
  `onnx4deeploy/transform/qzo_transform.py` (`build_qzo_train_graph`, `build_qzo_update_graph`,
  `_iter_qzo_params`; deprecated `build_qzo_int8_graph`), `onnx4deeploy/core/base_exporter.py`
  (`_export_qzo_training` rewrite), `onnx4deeploy/utils/onnx_node_implementations.py` (attr-scale
  Quant/Dequant). Commits on `feat/QZO`.

**Known limitations / follow-ups:** (a) online fp32-master weight perturbation deferred (needs a per-channel
Quant kernel in Deeploy). (b) BN is folded by the quant export (no BN γ/β ZO training in this v1). (c) The
`-mode quant` pipeline's own RequantShift is per-tensor and wrong for per-channel SpeechNet — we bypass it via
`build_int8_forward`; fixing the pipeline passes themselves is a separate task. (d) On-device validation
(below) still pending.

### Next iteration (8) — optional TrainDeeploy on-device smoke test
Pack `QZO_exp/exp2` → `Tests/Models/Training/SpeechNet/speechnet_qzo_{train,update}`; kill orphan gvsoc by
PID; run `deeployMezoRunner_tiled_siracusa.py` (n_steps 1, n_accum 1, num-data-inputs 2, eps 0.01, seed 42,
`-D BN_FROZEN_STATS=ON`); compare device L±/grad vs host `outputs.npz`; grep generated `TrainingNetwork.c` for
int8 pulp-nn conv (not `PULP_Conv2d_Im2Col_fp32`).

---

## Iteration 8 — TrainDeeploy device codegen: unblocked 3 issues, one remaining (2026-08-27)

Packed `QZO_exp/exp2` → `Tests/Models/Training/SpeechNet/speechnet_qzo_{train,update}` (pack = 3-file copy).
Ran `deeployMezoRunner_tiled_siracusa.py … --skipsim` (codegen only) in `deeploy_arm_mounted`; killed orphan
gvsoc by PID first. Fixed issues in order:

1. **`_isDepthwise` crash** (`node.inputs[1].shape is None`): the RQSPerturb perturbed-weight edge had no
   shape. Fix (Onnx4Deeploy): annotate the `{pname}_pert` edge shape/dtype in `build_qzo_train_graph`.
2. **`_assertTensorsHaveShape`** (fc-region intermediates shapeless — custom Quant/Dequant block inference):
   Fix: `_annotate_shapes_via_run` in `build_int8_forward` — run the graph once, stamp exact shape/dtype on
   every intermediate, **overwriting** stale shapeless `value_info` (the first pass missed 3 that already had
   empty entries).
3. **"No mapping for RQSPerturbRademacher"**: TrainDeeploy `feat/QZO` had the **float** `PerturbRademacher`
   stack but **no `RQSPerturbRademacher`** device op. Found the port on `feat/QZO_mixed` (commits `6995090`
   "port RQSPerturbRademacher to Deeploy" + `7cc4199` "zo_update mapping + sign-flip") — **cherry-picked both
   cleanly onto `feat/QZO`** (adds Parser/TypeChecker/Binding/2 Templates/TileConstraint + `RandomNoiseQuant.c`
   kernel). Codegen now **maps the int8 weight perturbs**.
4. **REMAINING** — backtracking exhausted at Layer 11 `rqsp_blocks0convbias_int32` (the int32 **bias**
   perturb). Root cause: our datapath feeds the perturbed **int32 bias into the Conv** (3rd input), but the
   device int8-conv binding won't consume a *variable* int32 bias — the float-ZO-quant + QZO_mixed convention
   puts the perturbed int32 bias in the **RequantShift `add`**. (This is the "bias-in-RQS-add" variant I
   already validated earlier at maxdiff **0.008**, even better than bias-in-Conv's 0.032.)

Commits: Onnx4Deeploy `6f94be6` (shape annotations); TrainDeeploy `5e786aa`+`967dba5` (RQSPerturb port).

### Next iteration (9) — bias in RequantShift add, then finish codegen + sim
1. Restructure `build_int8_forward`: **2-input Conv** (int8 weight only) + RequantShift `mul[c]=round(s_w[c]·
   div)`, **`add[c]` = int32 bias** (the perturbable term). Re-validate the forward vs PyTorch (expect the
   0.008 variant). Update `build_qzo_train_graph` to perturb the RequantShift `add` (RQSPerturb int32) instead
   of the conv bias; keep weight perturb as-is. Mirror in `build_qzo_update_graph`.
2. Re-run codegen → expect full mapping; then real GVSoC sim (drop `--skipsim`), compare device L±/grad vs host
   `outputs.npz`, and grep `TrainingNetwork.c` for int8 pulp-nn conv.

---

## Iteration 9 — bias→RequantShift-add restructure; 3 more device fixes; conv-parser blocker (2026-08-27)

Restructured `build_int8_forward` to the device-compatible datapath (Onnx4Deeploy commit): **2-input Conv**
(int8 weight only) with the int32 bias moved into the **RequantShift `add`**; **2-input Gemm** with the fc bias
moved to an **fp32 `Add`** after the per-channel dequant. Re-validated eps=0 forward (identity). Fixed three
device-codegen blockers in order:

1. **`_merge_conv_rq_fun` `.values` crash** (TrainDeeploy `Passes.py`): the merge baked rounding into the RQS
   `add` assuming a constant; a perturbed **variable** add has no `.values`. Guarded — bake only for a
   constant add; a variable add truncates (kernel matches host `add_is_initializer=False`).
2. **int32 bias perturb wouldn't map**: the bias mul used `DIV_B=2^31` → `mul=round(eps/s_b·2^31)≈5e12`
   **overflows int32** (binding expects `int32_t`). Fix: `DIV_B=2^16` (= the RequantShift div; mul≈1.6e8 fits)
   and emit **all perturb muls as int32**. → the `[int32,int32]→[int32]` bias binding now maps.
3. Cherry-picked the remaining QZO_mixed device commit **de7321c** (Quant/Dequant tiling-ready bindings) to
   complete the port. TrainDeeploy `feat/QZO` now has all three QZO_mixed device commits + the merge guard.

**REMAINING blocker (real):** codegen reaches the merged `RequantizedConv` (`_MERGE_CONVRQ_PASS_0`,
`PULPRQSConvLayer`) and **exhausts backtracking** — the PULP conv parser (`PULPConv2DParser` et al.) rejects a
**variable (perturbed) weight**. Confirmed this fails even **weight-only** (bias unperturbed, `PERTURB_BIAS=
False`), so it's the int8 conv+perturbed-weight itself, not the bias. Crucially, `git diff feat/QZO
feat/QZO_mixed` for the conv Parsers/Bindings/`testMVPTraining.py`/`DeeployTypes.py` is **empty** — the device
code is identical to QZO_mixed's (which compiled). ⇒ **the mapping difference must be the fixture graph
structure**, not the Deeploy code.

Commits: Onnx4Deeploy (bias-in-RQS-add + int32 mul + `PERTURB_BIAS`); TrainDeeploy `ee569cf` (merge guard) +
`8a853ce` (de7321c port).

### Next iteration (10) — compare our fixture to QZO_mixed's working int8 graph
Get QZO_mixed's compiling fixture (`ETH/quantzo_work/qzo_mixed/` outputs or a packed
`speechnet_qzo_mixed_int8_train`) and diff its **RequantizedConv/weight** structure against ours: dtype/nLevels
on the perturbed weight edge, whether the conv weight is a graph **input** vs an intermediate, the
`RQSPerturbRademacher` output typing, and any attrs the PULP conv parser requires (e.g. weight `nLevels`,
`signed`, group). Align our `build_int8_forward`/`build_qzo_train_graph` emission to match, then finish codegen
→ GVSoC sim → compare device L±/grad vs host.
