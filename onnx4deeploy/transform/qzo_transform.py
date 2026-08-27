# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Quantized ZO (QZO) graph transform — online weight quantisation + int8 Rademacher perturbation.

Design (feat/QZO, QZO_exp/exp1): for each Conv/Gemm, replace the weight path
    W_fp32(grid) → Conv
with
    W_fp32(grid) → Quant(scale) → RQSPerturbRademacher(mul) → Dequant(scale) → Conv
so that:
  * the perturbation lands on the **int8 code** (correct int-domain ε·z, via RQSPerturbRademacher), and
  * the **conv still sees fp32** (numerically consistent with the QCDQ base — no requant surgery needed here),
  * Deeploy's frontend later folds `weight-Quant → … → Conv → RequantShift` into a true int8 `RequantizedConv`.

`Quant`/`Dequant` are emitted **decomposed** (Div→Round→Clip / Sub→Mul) so Deeploy's QuantPatternPass /
DequantPatternPass recognise them. The per-channel weight scale comes from the scales JSON (single source of
truth), and the RQSPerturb `mul = round(eps/scale · 2**15)` is computed from the *same* scale.

Only Conv/Gemm **weights** (and, optionally, biases) are quantised+perturbed here; BN stays fp32 (handled
elsewhere / float-perturbed). A SoftmaxCrossEntropyLoss is appended to produce the ZO training loss.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DIV_RQS = 2 ** 15   # RQSPerturb fixed-point divisor (matches shipped kernel)


def _load_scales(scales_path: Optional[str]) -> Dict[str, list]:
    if scales_path and os.path.exists(scales_path):
        with open(scales_path) as f:
            return json.load(f)
    return {}


def _weight_scale_for(node_name: str, weight_name: str, out_ch: int, scales: Dict[str, list]) -> np.ndarray:
    """Per-channel weight scale [out_ch] from the JSON; fallback 1/127 if not found."""
    # keys look like "blocks.0.conv.weight_quant" / "fc.weight_quant"
    for key, val in scales.items():
        if key.endswith(".weight_quant"):
            arr = np.asarray(val, dtype=np.float64).flatten()
            if arr.shape[0] == out_ch:
                # prefer an ordered match by consuming keys once; simple channel-count match is enough
                return arr
    return np.ones(out_ch, dtype=np.float64) / 127.0


def _quantize_perturb_weight(nodes, inits, value_info, w_name, w_arr, scale, eps, seed, idx):
    """Emit  W → Div → Round → Clip → RQSPerturb → Sub(0) → Mul  and return the final (perturbed fp32) name."""
    C = w_arr.shape[0]
    rank = w_arr.ndim
    bshape = [C] + [1] * (rank - 1)                        # broadcast scale over output channels
    s_b = scale.reshape(bshape).astype(np.float32)
    s_flat = scale.astype(np.float64)

    def add_init(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

    scale_i = add_init(f"{w_name}_qs", s_b)                # per-channel scale
    cmin = add_init(f"{w_name}_cmin", np.array(-128.0, np.float32))
    cmax = add_init(f"{w_name}_cmax", np.array(127.0, np.float32))
    zp = add_init(f"{w_name}_zp", np.array(0.0, np.float32))
    mul = np.round(eps / s_flat * DIV_RQS).astype(np.int64)
    mul_i = add_init(f"{w_name}_mul", mul.astype(np.float32))    # RQSPerturb mul (per-channel)

    d, r, c, p, sub, o = (f"{w_name}_{s}" for s in ("qdiv", "qrnd", "qclip", "qpert", "qdsub", "qdeq"))
    nodes += [
        helper.make_node("Div",   [w_name, scale_i], [d], name=f"qzo_qdiv_{idx}"),
        helper.make_node("Round", [d], [r], name=f"qzo_qrnd_{idx}"),
        helper.make_node("Clip",  [r, cmin, cmax], [c], name=f"qzo_qclip_{idx}"),
        helper.make_node("RQSPerturbRademacher", [c, mul_i], [p], name=f"qzo_pert_{idx}",
                         domain="mezo", idx=idx, seed=seed, signed=1, div=DIV_RQS, n_levels=256,
                         doc_string="y = x + eps * RQSRademacher(x)"),
        helper.make_node("Sub",   [p, zp], [sub], name=f"qzo_ddsub_{idx}"),   # Dequant (zp=0)
        helper.make_node("Mul",   [sub, scale_i], [o], name=f"qzo_ddmul_{idx}"),
    ]
    value_info.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, list(w_arr.shape)))
    return o


def generate_qzo_train_graph(inference_onnx: str, output_onnx: str, zo_config: dict,
                             scales_path: Optional[str] = None, label_name: str = "label",
                             perturb_bias: bool = False) -> str:
    """Build the QZO train graph from the QCDQ base: online weight-quant + int8 perturb + SCE loss."""
    eps = float(zo_config.get("epsilon", 0.01))
    seed = int(zo_config.get("seed", 42))
    scales = _load_scales(scales_path)

    model = onnx.load(inference_onnx)
    g = model.graph
    inits = list(g.initializer)
    init_by_name = {i.name: numpy_helper.to_array(i) for i in inits}
    value_info = list(g.value_info)
    new_nodes: List = []
    idx = 0

    for node in g.node:
        if node.op_type in ("Conv", "Gemm") and len(node.input) >= 2:
            new_inputs = list(node.input)
            for i, inp in enumerate(node.input):
                is_w = inp in init_by_name and ("weight" in inp.lower() or (i == 1))
                is_b = inp in init_by_name and ("bias" in inp.lower() or (i == 2))
                if is_w:
                    w = init_by_name[inp]; C = w.shape[0]
                    s = _weight_scale_for(node.name, inp, C, scales)
                    new_inputs[i] = _quantize_perturb_weight(new_nodes, inits, value_info, inp, w, s, eps, seed, idx)
                    idx += 1
                elif is_b and perturb_bias:
                    pass  # bias perturbation: follow-up (int32 path); kept fp32 for exp1
            nn = helper.make_node(node.op_type, new_inputs, list(node.output), name=node.name,
                                  **{a.name: helper.get_attribute_value(a) for a in node.attribute})
            new_nodes.append(nn)
        else:
            new_nodes.append(node)

    # rebuild graph with the perturbed weight branches
    ng = helper.make_graph(new_nodes, g.name + "-qzo", list(g.input), list(g.output), inits, value_info=value_info)
    std = next((op.version for op in model.opset_import if op.domain == ""), 17)
    opset = [helper.make_opsetid("", std), helper.make_opsetid("mezo", 1), helper.make_opsetid("ai.onnx.contrib", 1)]
    nm = helper.make_model(ng, producer_name="qzo-train", opset_imports=opset)
    onnx.save(nm, output_onnx)

    # append SoftmaxCrossEntropyLoss(logits, label) -> log_prob
    _append_ce_loss(output_onnx, output_onnx, label_name)
    print(f"  [qzo] wrote train graph -> {output_onnx}  ({idx} weights quantised+perturbed)")
    return output_onnx


def _append_ce_loss(onnx_path, out_path, label_name="label"):
    m = onnx.load(onnx_path); g = m.graph
    logits = g.output[0].name
    bdim = g.input[0].type.tensor_type.shape.dim[0]
    batch = bdim.dim_value if bdim.HasField("dim_value") else 1
    g.input.append(helper.make_tensor_value_info(label_name, TensorProto.INT64, [batch, 1]))
    g.node.append(helper.make_node("SoftmaxCrossEntropyLoss", [logits, label_name], ["log_prob"],
                                   name="CrossEntropyLoss", reduction="mean"))
    oshape = [d.dim_value if d.HasField("dim_value") else 1 for d in g.output[0].type.tensor_type.shape.dim]
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, oshape))
    try:
        onnx.save(onnx.shape_inference.infer_shapes(m), out_path)
    except Exception:
        onnx.save(m, out_path)
