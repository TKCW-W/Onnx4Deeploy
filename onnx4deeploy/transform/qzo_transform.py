# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Quantized ZO (QZO) int8 datapath builder for SpeechNet — matches the exp6 device conventions.

Emits the TRUE int8 datapath with the SAME structural conventions as the shipped float-ZO device fixture
(exp6_ZO_single_step_latency): **trainable params are graph INPUTS** (not initializers), BN is
**BatchNormInternal** (com.microsoft, training_mode=1, 5 outputs), and every Quant is **Div→Add→Round→Clip**
(Add = zero_point, =0 but PRESENT) / Dequant is **Sub→Mul** (Sub = zero_point) so Deeploy's
QuantPatternPass / DequantPatternPass recognise them.

  per conv block:
     a_int8 → Conv(int8 act, int8 weight) → RequantShift(int32→int8, per-ch mul, int32 add=bias) → Dequant(s_out)
              → BatchNormInternal(fp32, unfolded) → ReLU → MaxPool → Quant(s_in_next) → a_int8_next
     weight:  W_fp32(INPUT) → Quant(Div→Add→Round→Clip, s_w per-ch) → RQSPerturbRademacher → int8 → Conv  (no dequant)
     bias:    bias_fp32(INPUT) → (·div/s_out, Round → int32) → RQSPerturbRademacher(int32) → RequantShift `add`
     BN γ/β:  gamma/beta(INPUT) → PerturbRademacher (float) → BatchNormInternal   (BN trains in fp32)
  head:  GAP → Quant → Gemm(int8) + int32 bias(perturbed) → per-class dequant → fp32 logits

Scale math (validated): conv mul[c]=round(s_in·s_w[c]/s_out·2¹⁶), weight-perturb mul=round(ε/s_w·2¹⁵),
bias add=round(bias/s_out·2¹⁶). Loss is NOT appended here — generate_zo_graph does that (append_cross_entropy_loss).
Returns (out_path, param_inputs) where param_inputs {name: np.ndarray} are the trainable INPUT values for inputs.npz.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIV_CONV = 2 ** 16     # RequantShift shift for convs
DIV_PERT = 2 ** 15     # RQSPerturb fixed-point divisor


def _np(x):
    return x.detach().cpu().float().numpy()


def build_qzo_int8_graph(model, scales, out_path, eps=0.01, seed=42, label_name="label"):
    """Build the int8 QZO graph (weights-as-inputs, Add/Sub, BatchNormInternal); NO loss (generate_zo_graph adds it)."""
    model.eval()
    nodes, inits, ginfo = [], [], []
    param_inputs = {}                      # trainable INPUT name -> value (for inputs.npz)
    pcount = [0]

    def K(name, arr):                      # non-trainable initializer (scales, bounds, running stats)
        inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

    def PARAM(name, arr, dtype=TensorProto.FLOAT):    # trainable graph INPUT
        a = np.asarray(arr, np.float32)
        ginfo.append(helper.make_tensor_value_info(name, dtype, list(a.shape)))
        param_inputs[name] = a
        return name

    def quant(x, s, tag, per_ch_shape=None, with_zp=True):
        """Quant = Div(scale) → [Add(zp=0)] → Round → Clip[-128,127]. s scalar or per-channel array.

        with_zp keeps the Add(zero_point) so Deeploy's QuantPatternPass recognises the full Div→Add→Round→Clip
        shape (per-TENSOR activations). Per-CHANNEL weight Quants set with_zp=False: QuantPatternPass is
        per-tensor only (does scale.item(), which raises on a size-C scale), and the weight Quant does NOT need
        to fold — PULPConvRequantMergePass passes conv weights through unchanged. zp=0 so dropping Add is a no-op.
        """
        s_arr = np.asarray(s, np.float32)
        if per_ch_shape is not None:
            s_arr = s_arr.reshape(per_ch_shape)
        d, ad, r, c = f"{tag}_qd", f"{tag}_qa", f"{tag}_qr", f"{tag}_q8"
        nodes.append(helper.make_node("Div", [x, K(f"{tag}_s", s_arr)], [d]))
        rnd_in = d
        if with_zp:
            nodes.append(helper.make_node("Add", [d, K(f"{tag}_zp", np.array(0., np.float32))], [ad]))  # zp=0 (present)
            rnd_in = ad
        nodes.append(helper.make_node("Round", [rnd_in], [r]))
        nodes.append(helper.make_node("Clip", [r, K(f"{tag}_lo", np.array(-128., np.float32)),
                                               K(f"{tag}_hi", np.array(127., np.float32))], [c]))
        return c

    def dequant(q, s, tag):
        """Dequant = Sub(zp=0) → Mul(scale). s scalar."""
        sb, m = f"{tag}_ds", f"{tag}_dm"
        nodes.append(helper.make_node("Sub", [q, K(f"{tag}_dzp", np.array(0., np.float32))], [sb]))   # zp=0 (present)
        nodes.append(helper.make_node("Mul", [sb, K(f"{tag}_dsc", np.array(s, np.float32))], [m]))
        return m

    def perturb_weight(W_name, s_w, W_shape, tag):
        """W_fp32(INPUT) → Quant(per-ch s_w) → RQSPerturbRademacher → int8 code."""
        C = W_shape[0]
        q8 = quant(W_name, s_w.astype(np.float32), f"{tag}_w",
                   per_ch_shape=[C] + [1] * (len(W_shape) - 1), with_zp=False)   # per-channel: no fold, no crash
        wmul = np.round(eps / s_w * DIV_PERT).astype(np.int64).astype(np.float32)
        p = f"{tag}_wp"
        nodes.append(helper.make_node("RQSPerturbRademacher", [q8, K(f"{tag}_wmul", wmul)], [p],
                     name=f"rqs_w_{tag}", domain="mezo", idx=pcount[0], seed=seed, signed=1,
                     div=DIV_PERT, n_levels=256))
        pcount[0] += 1
        return p

    def perturb_bias(b_name, s_out, div, tag):
        """bias_fp32(INPUT) → Mul(div/s_out) → Round → int32 code → RQSPerturbRademacher(int32) → RequantShift add."""
        sc, r = f"{tag}_bsc", f"{tag}_br"
        nodes.append(helper.make_node("Mul", [b_name, K(f"{tag}_bk", np.array(div / s_out, np.float32))], [sc]))
        nodes.append(helper.make_node("Round", [sc], [r]))
        bmul = np.array(round(eps * div / s_out * DIV_PERT))
        p = f"{tag}_bp"
        nodes.append(helper.make_node("RQSPerturbRademacher", [r, K(f"{tag}_bmul", np.full(1, bmul, np.float32))], [p],
                     name=f"rqs_b_{tag}", domain="mezo", idx=pcount[0], seed=seed, signed=1,
                     div=2 ** 31, n_levels=2 ** 32))
        pcount[0] += 1
        return p

    def perturb_bn(g_name, tag):
        """BN γ/β (fp32 INPUT) → PerturbRademacher (float ±ε) — BN trains in the fp32 path."""
        p = f"{tag}_bnp"
        nodes.append(helper.make_node("PerturbRademacher", [g_name], [p],
                     name=f"pert_{tag}", domain="mezo", idx=pcount[0], seed=seed, eps=eps))
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
        # trainable INPUTS
        Wn = PARAM(f"blocks.{bi}.conv.weight", W)
        bn_ = PARAM(f"blocks.{bi}.conv.bias", bias)
        gn = PARAM(f"blocks.{bi}.bn.weight", _np(bn.weight))
        ben = PARAM(f"blocks.{bi}.bn.bias", _np(bn.bias))
        # activation quant → int8
        a8 = quant(a, s_in, f"{tag}_ain")
        # weight quant+perturb, int8 conv (no conv bias — bias goes into requant add)
        Wp = perturb_weight(Wn, s_w, W.shape, tag)
        conv_out = f"{tag}_conv"
        nodes.append(helper.make_node("Conv", [a8, Wp], [conv_out], name=f"conv_{tag}",
                     dilations=list(conv.dilation), group=conv.groups, kernel_shape=list(conv.kernel_size),
                     pads=list(conv.padding) * 2, strides=list(conv.stride)))
        # requant int32→int8: mul per-channel, add = perturbed int32 bias
        mul = np.round(s_in * s_w / s_out * DIV_CONV).astype(np.int64).astype(np.float32)
        add_p = perturb_bias(bn_, s_out, DIV_CONV, tag)
        o8 = f"{tag}_o8"
        nodes.append(helper.make_node("RequantShift", [conv_out, K(f"{tag}_mul", mul), add_p], [o8],
                     name=f"rqs_{tag}", domain="ai.onnx.contrib", div=DIV_CONV, n_levels=256, signed=1))
        # dequant (Sub→Mul, s_out) → fp32 for BN
        odq = dequant(o8, s_out, f"{tag}_o")
        # BatchNormInternal (com.microsoft, 5 outputs, training_mode=1); γ/β perturbed float; μ/σ initializers
        gp = perturb_bn(gn, f"{tag}_g"); bep = perturb_bn(ben, f"{tag}_b")
        bn_o = f"{tag}_bn"
        nodes.append(helper.make_node("BatchNormInternal",
                     [odq, gp, bep, K(f"{tag}_m", _np(bn.running_mean)), K(f"{tag}_v", _np(bn.running_var))],
                     [bn_o, f"{tag}_rm", f"{tag}_rv", f"{tag}_sm", f"{tag}_sv"],
                     domain="com.microsoft", epsilon=float(bn.eps), momentum=float(bn.momentum or 0.0),
                     training_mode=1))
        relu_o = f"{tag}_relu"
        nodes.append(helper.make_node("Relu", [bn_o], [relu_o]))
        if hasattr(block.pool, "kernel_size"):
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
    Wfc = _np(model.fc.weight); bfc = _np(model.fc.bias)
    Wfcn = PARAM("fc.weight", Wfc); bfcn = PARAM("fc.bias", bfc)
    fc_in8 = quant(flat, s_in_fc, "fc_in")
    Wfc_p = perturb_weight(Wfcn, s_w_fc, Wfc.shape, "fc")
    gemm = "gemm"
    nodes.append(helper.make_node("Gemm", [fc_in8, Wfc_p], [gemm], name="gemm_fc", transB=1))
    s_bias = s_in_fc * s_w_fc                                    # per-class accumulator scale
    # fc bias → int32 code (per-class) → perturb → add
    fc_bsc = "fc_bsc"; fc_br = "fc_br"
    nodes.append(helper.make_node("Mul", [bfcn, K("fc_bk", (1.0 / s_bias).astype(np.float32))], [fc_bsc]))
    nodes.append(helper.make_node("Round", [fc_bsc], [fc_br]))
    bmul_fc = np.round(eps / s_bias * DIV_PERT).astype(np.int64).astype(np.float32)
    bp = "fc_bp"
    nodes.append(helper.make_node("RQSPerturbRademacher", [fc_br, K("fc_bmul", bmul_fc)], [bp],
                 name="rqs_b_fc", domain="mezo", idx=pcount[0], seed=seed, signed=1, div=2 ** 31, n_levels=2 ** 32))
    pcount[0] += 1
    summ = "fc_sum"
    nodes.append(helper.make_node("Add", [gemm, bp], [summ]))
    logits = "output"
    nodes.append(helper.make_node("Mul", [summ, K("fc_dqmul", s_bias.astype(np.float32))], [logits]))

    # Deeploy's reach algorithm dedups reachable nodes BY NAME (reachingSet uses {node.name}); unnamed nodes
    # all collide on "" so every topology pass consumes the whole graph. Give every node a unique name. -- QW
    for _i, _n in enumerate(nodes):
        if not _n.name:
            _n.name = f"{_n.op_type}_{_i}"

    g = helper.make_graph(nodes, "speechnet-qzo-int8",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                                         [1, 1, model.num_channels, model.time_steps])] + ginfo,
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, model.num_classes])],
                          inits)
    opset = [helper.make_opsetid("", 13), helper.make_opsetid("mezo", 1),
             helper.make_opsetid("ai.onnx.contrib", 1), helper.make_opsetid("com.microsoft", 1)]
    m = helper.make_model(g, producer_name="qzo-int8", opset_imports=opset)
    onnx.save(m, out_path)
    print(f"  [qzo] int8 datapath (weights-as-inputs, BatchNormInternal, Add/Sub) -> {out_path}  "
          f"({pcount[0]} perturb nodes, {len(param_inputs)} trainable inputs)")
    return out_path, param_inputs
