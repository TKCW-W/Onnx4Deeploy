# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Quantized ZO (QZO) int8 datapath builder for SpeechNet.

Emits the TRUE int8 datapath (not QCDQ-on-fp32) directly from a calibrated QuantSpeechNet + its scales:

  per conv block:
     a_int8 → Conv(int8 act, int8 weight) → RequantShift(int32→int8, per-ch mul, int32 add) → Dequant(s_out)
              → BatchNormalization(fp32, unfolded) → ReLU → MaxPool → Quant(s_in_next) → a_int8_next
     weight:  W_fp32 → Quant(s_w, per-ch) → RQSPerturbRademacher → int8 → Conv          (NO weight dequant)
     bias:    add = round(bias/s_out · div)  (int32, the RequantShift 3rd input) → RQSPerturbRademacher(int32)

  head:  GAP → Quant(s_in_fc) → Gemm(int8) + int32 bias(perturbed) → per-class dequant → fp32 logits
  loss:  SoftmaxCrossEntropyLoss(logits, label) → log_prob

Scales come from the scales JSON (single source of truth): mul[c] = round(s_in·s_w[c]/s_out · div),
RQSPerturb weight mul = round(ε/s_w · 2¹⁵), bias mul = round(ε·div/s_out · 2¹⁵).
Quant is emitted decomposed (Div→Round→Clip) and activation Dequant per-tensor (Mul) so Deeploy folds them;
Conv→RequantShift folds to RequantizedConv. All int tensors are fp32-valued integers (QCDQ convention that
run_onnx_graph executes).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIV_CONV = 2 ** 16     # RequantShift shift for convs (mul fits int, x*mul stays in int32)
DIV_PERT = 2 ** 15     # RQSPerturb fixed-point divisor


def _np(x):
    return x.detach().cpu().float().numpy()


def build_qzo_int8_graph(model, scales, out_path, eps=0.01, seed=42, label_name="label"):
    """Construct the int8 QZO train graph from a calibrated QuantSpeechNet + scales dict. Returns out_path."""
    model.eval()
    nodes, inits, vinfo = [], [], []
    pcount = [0]

    def K(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

    def quant_act(x, s, tag):
        """fp32 x → int8 (Div s → Round → Clip[-128,127]); per-tensor scalar s."""
        d, r, c = f"{tag}_qd", f"{tag}_qr", f"{tag}_q8"
        nodes.append(helper.make_node("Div", [x, K(f"{tag}_s", np.array(s, np.float32))], [d]))
        nodes.append(helper.make_node("Round", [d], [r]))
        nodes.append(helper.make_node("Clip", [r, K(f"{tag}_lo", np.array(-128., np.float32)),
                                               K(f"{tag}_hi", np.array(127., np.float32))], [c]))
        return c

    def quant_perturb_weight(W, s_w, tag):
        """W_fp32(init) → Div(per-ch s_w) → Round → Clip → RQSPerturb → int8 code (fp32-valued)."""
        C = W.shape[0]
        s_b = s_w.reshape([C] + [1] * (W.ndim - 1)).astype(np.float32)
        wname = K(f"{tag}_W", W.astype(np.float32))
        d, r, c, p = f"{tag}_wd", f"{tag}_wr", f"{tag}_w8", f"{tag}_wp"
        nodes.append(helper.make_node("Div", [wname, K(f"{tag}_ws", s_b)], [d]))
        nodes.append(helper.make_node("Round", [d], [r]))
        nodes.append(helper.make_node("Clip", [r, K(f"{tag}_wlo", np.array(-128., np.float32)),
                                               K(f"{tag}_whi", np.array(127., np.float32))], [c]))
        wmul = np.round(eps / s_w * DIV_PERT).astype(np.int64).astype(np.float32)
        nodes.append(helper.make_node("RQSPerturbRademacher", [c, K(f"{tag}_wmul", wmul)], [p],
                     name=f"rqs_w_{tag}", domain="mezo", idx=pcount[0], seed=seed, signed=1,
                     div=DIV_PERT, n_levels=256))
        pcount[0] += 1
        return p

    def perturb_bias_i32(bias, s_out, div, tag):
        """add[c] = round(bias/s_out · div) int32, perturbed by its own int32 RQSPerturb."""
        add = np.round(bias / s_out * div).astype(np.int64).astype(np.float32)
        bmul = np.round(eps * div / s_out * DIV_PERT).astype(np.int64).astype(np.float32)
        p = f"{tag}_bp"
        nodes.append(helper.make_node("RQSPerturbRademacher", [K(f"{tag}_add", add), K(f"{tag}_bmul", bmul)], [p],
                     name=f"rqs_b_{tag}", domain="mezo", idx=pcount[0], seed=seed, signed=1,
                     div=2 ** 31, n_levels=2 ** 32))
        pcount[0] += 1
        return p

    a = "input"
    for bi, block in enumerate(model.blocks):
        conv, bn = block.conv, block.bn
        tag = f"b{bi}"
        s_in = float(scales[f"blocks.{bi}.conv.input_quant"])
        s_out = float(scales[f"blocks.{bi}.conv.output_quant"])
        s_w = np.asarray(scales[f"blocks.{bi}.conv.weight_quant"], np.float64)
        W = _np(conv.weight); bias = _np(conv.bias)
        Cout = W.shape[0]
        # activation quant
        a8 = quant_act(a, s_in, f"{tag}_ain")
        # weight quant+perturb
        Wp = quant_perturb_weight(W, s_w, tag)
        # int8 conv (fp32-valued int), no bias in the conv (bias goes into the requant add)
        conv_out = f"{tag}_conv"
        katt = dict(dilations=list(conv.dilation), group=conv.groups,
                    kernel_shape=list(conv.kernel_size), pads=list(conv.padding) * 2, strides=list(conv.stride))
        nodes.append(helper.make_node("Conv", [a8, Wp], [conv_out], name=f"conv_{tag}", **katt))
        # requant: mul = round(s_in·s_w/s_out · div), add = perturbed int32 bias
        mul = np.round(s_in * s_w / s_out * DIV_CONV).astype(np.int64).astype(np.float32)
        add_p = perturb_bias_i32(bias, s_out, DIV_CONV, tag)
        o8 = f"{tag}_o8"
        nodes.append(helper.make_node("RequantShift", [conv_out, K(f"{tag}_mul", mul), add_p], [o8],
                     name=f"rqs_{tag}", domain="ai.onnx.contrib", div=DIV_CONV, n_levels=256, signed=1))
        # dequant (per-tensor s_out) → fp32 for BN
        odq = f"{tag}_odq"
        nodes.append(helper.make_node("Mul", [o8, K(f"{tag}_sout", np.array(s_out, np.float32))], [odq]))
        # unfolded fp32 BN
        bn_o = f"{tag}_bn"
        nodes.append(helper.make_node("BatchNormalization",
                     [odq, K(f"{tag}_g", _np(bn.weight)), K(f"{tag}_be", _np(bn.bias)),
                      K(f"{tag}_m", _np(bn.running_mean)), K(f"{tag}_v", _np(bn.running_var))],
                     [bn_o], epsilon=float(bn.eps)))
        relu_o = f"{tag}_relu"
        nodes.append(helper.make_node("Relu", [bn_o], [relu_o]))
        if isinstance(block.pool, type(model.blocks[0].pool)) and hasattr(block.pool, "kernel_size"):
            po = f"{tag}_pool"
            nodes.append(helper.make_node("MaxPool", [relu_o], [po],
                         kernel_shape=list(block.pool.kernel_size), strides=list(block.pool.stride)))
            a = po
        else:
            a = relu_o

    # head: GAP → int8 → Gemm(int8) + int32 bias → per-class dequant → fp32 logits
    gap = "gap"
    nodes.append(helper.make_node("GlobalAveragePool", [a], [gap]))
    flat = "flat"
    nodes.append(helper.make_node("Reshape", [gap, K("flat_shape", np.array([1, model._fc_in], np.int64))], [flat]))
    s_in_fc = float(scales["fc.input_quant"]); s_w_fc = np.asarray(scales["fc.weight_quant"], np.float64)
    fc_in8 = quant_act(flat, s_in_fc, "fc_in")
    Wfc = _np(model.fc.weight)                       # [num_classes, fc_in]
    Wfc_p = quant_perturb_weight(Wfc, s_w_fc, "fc")
    gemm = "gemm"
    nodes.append(helper.make_node("Gemm", [fc_in8, Wfc_p], [gemm], transB=1))   # int8 gemm, no bias here
    # bias in accumulator domain (s_in_fc·s_w_fc), perturbed int32
    bias_fc = _np(model.fc.bias)
    s_bias = s_in_fc * s_w_fc
    add_fc = np.round(bias_fc / s_bias).astype(np.int64).astype(np.float32)
    bmul_fc = np.round(eps / s_bias * DIV_PERT).astype(np.int64).astype(np.float32)
    bp = "fc_bp"
    nodes.append(helper.make_node("RQSPerturbRademacher", [K("fc_addv", add_fc), K("fc_bmul", bmul_fc)], [bp],
                 name="rqs_b_fc", domain="mezo", idx=pcount[0], seed=seed, signed=1, div=2 ** 31, n_levels=2 ** 32))
    pcount[0] += 1
    summ = "fc_sum"
    nodes.append(helper.make_node("Add", [gemm, bp], [summ]))
    logits = "output"
    dqmul = (s_in_fc * s_w_fc).astype(np.float32)     # per-class dequant to fp32 logits
    nodes.append(helper.make_node("Mul", [summ, K("fc_dqmul", dqmul)], [logits]))

    # loss
    nodes.append(helper.make_node("SoftmaxCrossEntropyLoss", [logits, label_name], ["log_prob"],
                 name="CrossEntropyLoss", reduction="mean"))

    graph = helper.make_graph(
        nodes, "speechnet-qzo-int8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, model.num_channels, model.time_steps]),
         helper.make_tensor_value_info(label_name, TensorProto.INT64, [1, 1])],
        [helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, [1, model.num_classes])],
        inits)
    opset = [helper.make_opsetid("", 13), helper.make_opsetid("mezo", 1), helper.make_opsetid("ai.onnx.contrib", 1)]
    m = helper.make_model(graph, producer_name="qzo-int8", opset_imports=opset)
    onnx.save(m, out_path)
    print(f"  [qzo] int8 datapath -> {out_path}  ({pcount[0]} RQSPerturb nodes)")
    return out_path
