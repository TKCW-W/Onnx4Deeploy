# QZO exp1 — SpeechNet int8 quantized-ZO: implemented flow (verified)

**Date:** 2026-08-27 · **Repo/branch:** `Onnx4Deeploy` `feat/QZO` · Clean build (no reuse of `feat/QZO_mixed`).
**CLI (extension on `Onnx4Deeploy.py`, not a side pipeline):**
`python Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher -o QZO_exp/exp1`
**Quant scope:** int8 **Conv weights + biases** and **FC weight + bias**; **BatchNorm unfolded fp32** (γ/β
trained in the fp32 path).

## Datapath emitted — the *real* int8 path (not QCDQ-on-fp32), per conv block
```
a_int8 → Conv(int8 act, int8 weight) → RequantShift(int32→int8, per-ch mul=s_in·s_w/s_out, int32 add=bias)
       → Dequant(s_out) → BatchNormalization(fp32, UNFOLDED) → ReLU → MaxPool → Quant(s_in_next) → a_int8_next
  weight:  W_fp32 → Quant(s_w per-ch) → RQSPerturbRademacher → int8 → Conv        (NO weight dequant)
  bias:    RequantShift int32 `add` = round(bias/s_out·div), perturbed by its OWN int32 RQSPerturb (div=2³¹)
  head:    GAP → Quant(s_in_fc) → Gemm(int8) + int32 bias(perturbed) → per-class dequant → fp32 logits
  loss:    SoftmaxCrossEntropyLoss(logits, label) → log_prob
```
**Single input Quant** (block-0 `conv.input_quant`; the standalone `QuantIdentity` was removed — no more double
input-quant). Weights fed as fp32 → quantised online → int8 code → perturbed (int-domain ε·z). Scales come
from `speechnet_scales.json` (single source of truth): conv `mul[c]=round(s_in·s_w[c]/s_out·2¹⁶)`,
weight-perturb `mul=round(ε/s_w·2¹⁵)`, bias-perturb accordingly.

## Verification — ALL PASS (independently re-run, actual numbers)
| # | check | result |
|--|--|--|
| 1 | CLI `-mode q-zo-train` runs end-to-end | ✅ → network_zo_train.onnx, inputs/outputs.npz, speechnet_scales.json |
| 2 | single input Quant; Conv consumes int8; `Conv→RequantShift→Dequant→BN` | ✅ (input→`Div,Round,Clip,Conv`; 1st Conv→RequantShift) |
| 3 | exactly **12** `RQSPerturbRademacher` (6 weight + 6 bias) | ✅ |
| 4 | `run_onnx_graph` executes → log_prob[1,9] | ✅ (ref argmax varies by seed) |
| 5 | **`PULPOptimizer` fold** | ✅ **RequantizedConv=5, float Conv=0, RQSPerturb=12 preserved, BatchNormalization(fp32)=5, Gemm=1** |
| 6 | numerical sanity: int8(ε=0) argmax == Brevitas argmax (same input) | ✅ **MATCH** (restructuring preserved numerics) |

**Op histogram (pre-fold):** `Conv 5, RequantShift 5, RQSPerturbRademacher 12, BatchNormalization 5, Relu 5,
MaxPool 3, Div/Round/Clip 12, Mul 6, GlobalAveragePool 1, Reshape 1, Gemm 1, Add 1, SoftmaxCrossEntropyLoss 1`.
**Post-`PULPOptimizer`:** `RequantizedConv 5, BatchNormalization(fp32) 5, RQSPerturbRademacher 12, Gemm 1`
(+ activation Div/Round/Clip/Mul, GAP, Reshape, SCE).

**Honest caveat:** the FC head stays a **float `Gemm`** — its weight+bias are quantised+perturbed (int8), but the
fp32-logit head doesn't fold to `RequantizedGemm` (its output is fp32 for the loss). The **5 int8
`RequantizedConv` are the acceleration target**, so this is acceptable — flagged, not papered over.

## Code (extension on Onnx4Deeploy, `feat/QZO`)
- `onnx4deeploy/models/pytorch_models/speechnet/speechnet_quant.py` — `QuantSpeechNetDeploy`: int8 conv+fc
  (per-channel weights, int32 bias, int8 acts), conv-output observers (B1 fix), unfolded fp32 BN, **single
  input quant**.
- `onnx4deeploy/transform/quant_scale_dump.py` — `dump_brevitas_scales` → `speechnet_scales.json`
  (per-channel `weight_quant`, per-tensor `input_quant`/`output_quant`).
- `onnx4deeploy/transform/qzo_transform.py` — `build_qzo_int8_graph`: constructs the int8 datapath from the
  calibrated model + scales.
- `onnx4deeploy/models/speechnet_exporter.py` — `create_brevitas_model` + `get_calibration_data`.
- `onnx4deeploy/core/base_exporter.py` — `_export_qzo_training` (branched from `export_zo_training(quant=True)`).
- `Onnx4Deeploy.py` — `q-zo-train` CLI mode.

## Artifacts (this dir)
`generate.py` (standalone regenerator + sanity check), `network_zo_train.onnx`, `speechnet_scales.json`,
`inputs.npz`, `outputs.npz`.

## Next
- `zo_update` graph (fp32-master update: conv/fc via int8 direction; BN γ/β fp32 float-perturb).
- Real pretrained SpeechNet weights + SilentWear PTQ calibration (`--pretrained-weights`, `--dataset silentwear`).
- On-device single-step validation (device == host).
- `RequantShift div` fixed at 2¹⁶ — revisit per-conv if a layer saturates on real data.
- (Optional) FC → `RequantizedGemm` if the head should also be int8-accelerated.
