# QZO exp1 — SpeechNet int8 quantized-ZO (extends the existing ZO pipeline)

**Date:** 2026-08-27 · **Repo/branch:** `Onnx4Deeploy` `feat/QZO` · Clean build (no `feat/QZO_mixed` reuse).
**CLI (extension on `Onnx4Deeploy.py`, routed through the existing `generate_zo_graph`):**
`python Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher -o QZO_exp/exp1`
**Quant scope:** int8 **Conv weights+biases** and **FC weight+bias**; **BN unfolded fp32**, γ/β trained in the
fp32 path.

## Conventions match the device reference (exp6_ZO_single_step_latency/fixture/network_zo_train.onnx)
- **Trainable params are graph INPUTS, not initializers** — 24 inputs: `input, label, blocks.{0..4}.conv.
  weight/bias, blocks.{0..4}.bn.weight/bias (γ/β), fc.weight/bias` (22 trainable, matches exp6's 24).
- **`BatchNormInternal`** (domain `com.microsoft`, `training_mode=1`, 5 outputs) — unfolded, fp32.
- **Quant = `Div→Add→Round→Clip`, Dequant = `Sub→Mul`** (zero-point `Add`/`Sub` present, =0) so Deeploy's
  `QuantPatternPass`/`DequantPatternPass` recognise them.
- Produced **through `generate_zo_graph`** (extended with a `qzo_model`/`qzo_scales` branch), reusing
  `_promote_initializers_to_inputs`, the BN→BatchNormInternal path, and `append_cross_entropy_loss` — **not** a
  parallel builder.

## Datapath (per conv block)
```
a_int8 → Conv(int8) → RequantShift(int32→int8, per-ch mul=s_in·s_w/s_out, int32 add=bias)
       → Dequant(Sub→Mul, s_out) → BatchNormInternal(fp32) → ReLU → MaxPool → Quant(Div→Add→Round→Clip) → …
  weight (INPUT): W_fp32 → Quant(Div→Round→Clip, per-ch s_w) → RQSPerturbRademacher → int8 → Conv  (no dequant)
  bias   (INPUT): RequantShift int32 `add`, perturbed by its own int32 RQSPerturbRademacher
  BN γ/β (INPUT): fp32, perturbed by float PerturbRademacher (BN trains in fp32)
  head: GAP → Quant → Gemm(int8 w/b) → per-class dequant → fp32 logits → SoftmaxCrossEntropyLoss
```
**22 perturb nodes:** 12 `RQSPerturbRademacher` (int8 conv/fc weights+biases) + 10 float `PerturbRademacher`
(BN γ/β).

## Verification — ALL PASS (independently re-run, actual numbers)
| # | check | result |
|--|--|--|
| A | trainable params as INPUTS | ✅ 24 inputs; 0 trainable initializers (matches exp6) |
| B | Quant `Div→Add→Round→Clip`, Dequant `Sub→Mul` | ✅ Add=7, Sub=5 present |
| C | `BatchNormInternal` | ✅ com.microsoft, 5 outputs, training_mode=1 |
| D | produced via `generate_zo_graph` | ✅ `_export_qzo_training` + `generate.py` route through it |
| E | `run_onnx_graph` executes | ✅ → outputs.npz |
| F | **`PULPOptimizer` fold** | ✅ **RequantizedConv=5, float Conv=0, RQSPerturb=12 preserved, BatchNormInternal=5, float PerturbRademacher=10**; activation **Quant 0→6, Dequant 0→5** now recognised (Add/Sub fix) |
| G | numerical sanity int8(ε=0) argmax == Brevitas argmax | ✅ **3 == 3 MATCH** |

**Fold, before/after (point-B proof):** pre `Div12 Add7 Round18 Clip12 Sub5 Mul12 | Conv5 RequantShift5
RQSPerturb12 BatchNormInternal5 PerturbRademacher10 Gemm1`; post `RequantizedConv5 Quant6 Dequant5
RQSPerturb12 BatchNormInternal5 PerturbRademacher10 Gemm1` (+ raw per-channel weight Div/Round/Clip 6, bias
Mul, fc tail). The previous (Add/Sub-less) graph had `Quant=0, Dequant=0` — the activation quant stayed raw.

## Honest caveats (flagged, not papered over)
- **Per-channel weight Quant does NOT fold** to a `Quant` node — Deeploy's `QuantPatternPass` is per-tensor
  (`scale.item()`). The weight still enters `RequantizedConv` correctly (fold passes conv weights through), so
  it's fine for deployment; it just stays as raw `Div/Round/Clip` on device. Whether Deeploy should gain
  per-channel weight-Quant folding is a supervisor question.
- **FC head stays float `Gemm`** (fp32 logits for the loss); the 5 int8 `RequantizedConv` are the acceleration
  target.

## Code (extension on Onnx4Deeploy, `feat/QZO`)
- `models/pytorch_models/speechnet/speechnet_quant.py` — QuantSpeechNet (int8 conv+fc, conv-output observers,
  unfolded BN, single input quant).
- `transform/quant_scale_dump.py` — per-channel weight + per-tensor act scales → `speechnet_scales.json`.
- `transform/qzo_transform.py` — `build_qzo_int8_graph` (int8 datapath: weights-as-inputs, BatchNormInternal,
  Add/Sub Quant/Dequant, unique node names).
- `transform/zo_transform.py` — `generate_zo_graph` gained a `qzo_model`/`qzo_scales` branch (routes QZO
  through the shared pipeline + `append_cross_entropy_loss`).
- `models/speechnet_exporter.py` — `create_brevitas_model` + `get_calibration_data`.
- `core/base_exporter.py` — `_export_qzo_training` routes through `generate_zo_graph`, feeds the trainable input
  values into `inputs.npz`.
- `Onnx4Deeploy.py` — `q-zo-train` CLI mode.

## Artifacts: `generate.py`, `fold_check.py`, `network_zo_train.onnx`, `speechnet_scales.json`, `inputs.npz`, `outputs.npz`.

## Next
- `zo_update` graph (fp32-master update; conv/fc int8 direction, BN γ/β fp32 float-perturb) — mirror the
  existing `generate_weight_update_graph`.
- Real pretrained SpeechNet weights + SilentWear PTQ calibration (`--pretrained-weights`, `--dataset silentwear`).
- On-device single-step validation (device == host) in TrainDeeploy.
- (Supervisor) per-channel weight-Quant folding in Deeploy; optional FC → `RequantizedGemm`.
