# QZO exp1 — quantized-ZO SpeechNet extension: implemented flow

**Date:** 2026-08-27 · **Repo/branch:** `Onnx4Deeploy` `feat/QZO` · **Scope:** clean, first-principles build
(no reuse of the previous `feat/QZO_mixed` trial). Follows `ETH/docs/QZO_IMPLEMENTATION_GUIDE.md`.

**Quantization scope (as specified):** int8 for **Conv weights** and the **FC/GEMM weights**; **BatchNorm stays
unfolded in fp32** (γ/β trained in the fp32 path). Activations are int8 at the quant boundaries.

---

## 1. What was implemented (3 new modules on `feat/QZO`)

| file | role |
|---|---|
| `onnx4deeploy/models/pytorch_models/speechnet/speechnet_quant.py` | **QuantSpeechNet** — INT8 Brevitas SpeechNet: per-channel int8 conv/fc weights, int32 bias, int8 activations **with a conv-output observer** (`output_quant`, the B1 fix), **plain fp32 `nn.BatchNorm2d` (unfolded)**. |
| `onnx4deeploy/transform/quant_scale_dump.py` | **scale-dump** — writes the per-channel weight scales (+ act/bias scales) to `speechnet_scales.json` in the flat `<layer>.weight_quant` format `_get_weight_scale` expects. *The single source of truth* for the `Quant` scale and the `RQSPerturb mul`. (No shipped producer existed — this is it.) |
| `onnx4deeploy/transform/qzo_transform.py` | **QZO transform** — online weight-quant + int8 Rademacher perturbation: for each Conv/Gemm weight it emits `W_fp32 → Div→Round→Clip (Quant) → RQSPerturbRademacher(mul=round(ε/scale·2¹⁵)) → Sub→Mul (Dequant) → Conv`, then appends `SoftmaxCrossEntropyLoss`. |

## 2. The flow (reproduce with `QZO_exp/exp1/generate.py`)
```
QuantSpeechNet (Brevitas)  →  PTQ calibrate  →  dump per-channel scales (speechnet_scales.json)
   →  Export4Deeploy.exportBrevitas  →  network_infer.onnx  (QCDQ base: int8 convs, 5 UNFOLDED fp32 BN)
   →  qzo_transform  →  network_zo_train.onnx  (online weight-quant + int8 RQSPerturb + SCE loss)
   →  run_onnx_graph reference  →  inputs.npz / outputs.npz
```
Run: `PYTHONPATH=/app/Onnx4Deeploy:/app/Onnx4Deeploy/DeepQuant python3 generate.py` (in `agitated_hugle`).

## 3. What is validated ✅
- **QuantSpeechNet builds, calibrates, and exports** to the intended mixed QCDQ: 5 int8 `Conv`, **5 unfolded
  fp32 `BatchNormalization`**, ReLU/MaxPool, int8 `Gemm`, with conv-output observers (13 activation quant
  boundaries).
- **Scales dumped** correctly: 6 per-channel `weight_quant` entries (lengths 8/16/16/32/32/9) + act/bias scales.
- **The QZO train graph is produced and numerically executes**: `network_zo_train.onnx` has **6
  `RQSPerturbRademacher`** (5 conv + fc weights) each fed by an online weight `Quant`, plus
  `SoftmaxCrossEntropyLoss`. `run_onnx_graph` runs it end-to-end → `outputs.npz` = `log_prob[1,9]` ≈ −2.2
  (≈ log(1/9), sensible for random weights).
- **The perturbation is in the int8 domain** (RQSPerturb casts to int; the weight `Quant` gives it real int8
  codes) and the conv still sees fp32 (the weight `Dequant` restores the grid value) — numerically consistent
  with the QCDQ base.

## 4. Honest result on the Deeploy int8 fold ⚠️ (the expected blocker)
Running the shipped `PULPOptimizer` (zo-support) on `network_zo_train.onnx`: the convs **stay float**
(`Conv=5, RequantizedConv=0`), though `RQSPerturbRademacher` (6) and the fp32 `BatchNormalization` (5) are
preserved. Two causes, both anticipated in the guide:
1. **Unfolded BN blocks the conv fold.** With BN kept fp32, each conv output must **`Dequant` to fp32** for the
   BN — so there is **no `Conv → RequantShift`** for `PULPConvRequantMergePass` to merge into a
   `RequantizedConv`. (In the shipped *BN-free* QMCUNetZO fixture, the conv output goes straight to a
   `RequantShift`, which is why it folds to 42 `RequantizedConv`.)
2. **Per-channel weight Dequant** isn't recognised by `DequantPatternPass` (it does `.item()` on the scale,
   i.e. expects a per-**tensor** scalar). The per-channel weight scale must live in a **`RequantShift`**, not a
   weight `Dequant` node.

**Both are fixed by the same restructuring** (the documented next step): rewrite each conv as
`int8-Conv → RequantShift(int32→int8, per-channel mul = s_in·s_w/s_out) → Dequant(int8→fp32) → BN(fp32)`.
This (a) creates the `Conv→RequantShift` the merge needs, and (b) moves the per-channel `s_w` into the
`RequantShift` where Deeploy expects it. It is the `Conv→RequantShift→Dequant→BN` restructuring we scoped —
**exp2**.

## 5. Interpretation
Per the memory rule "*first step = correct compilable graph + fixture in Onnx4Deeploy*", **exp1 is the
Onnx4Deeploy stage done**: a clean, numerically-executing quantized-ZO SpeechNet graph with unfolded fp32 BN,
int8 online-quant conv+fc weights, int8 `RQSPerturbRademacher`, the SCE loss, and the scales JSON. The **int8
`RequantizedConv` acceleration is a Deeploy-stage concern** that requires the conv→RequantShift restructuring —
the natural exp2. This is exactly the "int8-conv vs unfolded-fp32-BN" tension we mapped: it is resolvable by
restructuring, not fundamental.

## 6. Artifacts (this dir)
- `generate.py` — end-to-end generator.
- `speechnet_scales.json` — per-channel weight scales (single source of truth).
- `network_infer.onnx` — QCDQ base (int8 convs + unfolded fp32 BN).
- `network_zo_train.onnx` — QZO train graph (online weight-quant + int8 RQSPerturb + SCE loss).
- `inputs.npz` (input + label), `outputs.npz` (log_prob reference).

## 7. Next (exp2)
1. Restructure `Conv→RequantShift→Dequant→BN` (per-channel `s_w` into the `RequantShift`) so `PULPOptimizer`
   yields `RequantizedConv` with `RQSPerturb` preserved (target: convs → RequantizedConv, 0 float Conv).
2. Add the `zo_update` graph (fp32-master update; conv/fc via the int8 direction, BN γ/β fp32 float-perturb).
3. Wire into `Onnx4Deeploy.py -mode q-zo-train` (create_brevitas_model + config) for one-command generation.
4. Pretrained SpeechNet weights + on-device single-step validation (device == host).
