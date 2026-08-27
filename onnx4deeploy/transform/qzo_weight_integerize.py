# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Per-channel weight/bias integerization for the quantized-ZO (QZO) datapath.

WHY THIS EXISTS
---------------
`create_quant_pipeline` (onnx4deeploy/core/optimization_passes.py) integerizes only
*per-tensor* QCDQ: `fold_qcdq_to_quant_dequant` skips any Div/Add/Round/Clip whose scale
is not a scalar (``_const_scalar`` returns None for ``size != 1``). SpeechNet's Brevitas
conv/fc use **per-channel** weight scales (``Int8WeightPerChannelFloat``), so their weight
QCDQ never folds — the exported `network.onnx` keeps f32 conv weights behind a
`Div→Add→Round→Clip→Sub→Mul→Conv` chain and the device would run an fp32 conv.

This module closes that gap for the QZO path: it evaluates each Conv/Gemm weight & bias
QCDQ chain at export time (per-channel), replaces the weight/bias with the int8 / int32
initializer feeding the consumer DIRECTLY, deletes the dead QCDQ+dequant nodes, and
returns the per-channel scale map so the caller can build the matching per-channel
post-conv RequantShift (``mul[c] = s_in·s_w[c]/s_out``) — Deeploy's RequantShift already
supports a per-channel ``mul`` vector (the general, non-Uniform template).

STATUS
------
- `trace_weight_qcdq` + int8/int32 evaluation + rewire: VERIFIED on the real SpeechNet
  `-mode quant` `network.onnx` (5 conv weights + fc → int8 [-127,127], per-channel s_w of
  size = out-channels; biases → int32 per-channel).
- Building the per-channel RequantShift from the returned scale map is done by the caller
  (QZO export path) — see Report.md "Next iteration".
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


def _build_maps(graph: onnx.GraphProto):
    prod = {o: n for n in graph.node for o in n.output}
    initmap = {i.name: i for i in graph.initializer}
    return prod, initmap


def _const_value(name: str, prod, initmap) -> Optional[np.ndarray]:
    """Resolve a constant ndarray for an initializer or a Constant/Cast/Identity chain."""
    if name in initmap:
        return numpy_helper.to_array(initmap[name])
    pn = prod.get(name)
    guard = 0
    while pn is not None and guard < 6:
        guard += 1
        if pn.op_type == "Constant":
            t = [a.t for a in pn.attribute if a.name == "value"]
            return numpy_helper.to_array(t[0]) if t else None
        if pn.op_type in ("Cast", "Identity") and pn.input:
            return _const_value(pn.input[0], prod, initmap)
        return None
    return None


def trace_weight_qcdq(start: str, prod, initmap) -> Optional[dict]:
    """From a Conv/Gemm weight or bias input, walk back the dequant+quant chain
    ``Mul <- Sub <- Clip <- Round <- Add <- Div <- source-initializer`` and return
    ``dict(src, W, s, zp, lo, hi, chain_nodes)`` or None if the shape isn't matched.
    Scales/zero-points/bounds may be initializers or Constant(/Cast) outputs.
    """
    def is_const(nm):
        return _const_value(nm, prod, initmap) is not None

    node = prod.get(start)
    guard = 0
    scale = zp = lo = hi = src = None
    chain_nodes: List[str] = []
    while node is not None and guard < 14:
        guard += 1
        chain_nodes.append(node.name)
        t = node.op_type
        if t == "Div":
            ins = list(node.input)
            src_cand = [i for i in ins if i in initmap]
            if src_cand:
                src = src_cand[0]
                scale = _const_value([i for i in ins if i != src][0], prod, initmap)
            else:
                consts = [(i, _const_value(i, prod, initmap)) for i in ins]
                consts = [(i, v) for i, v in consts if v is not None]
                consts.sort(key=lambda kv: kv[1].size)  # smaller == scale
                if len(consts) == 2:
                    scale, src = consts[0][1], consts[1][0]
                elif len(consts) == 1:
                    scale = consts[0][1]
                    src = [i for i in ins if i != consts[0][0]][0]
            break
        if t == "Add":
            for inp in node.input:
                v = _const_value(inp, prod, initmap)
                if v is not None:
                    zp = v
            nxt = [i for i in node.input if not is_const(i)]
            node = prod.get(nxt[0]) if nxt else None
            continue
        if t == "Clip":
            if len(node.input) >= 3:
                lv = _const_value(node.input[1], prod, initmap)
                hv = _const_value(node.input[2], prod, initmap)
                if lv is not None:
                    lo = float(np.asarray(lv).flatten()[0])
                if hv is not None:
                    hi = float(np.asarray(hv).flatten()[0])
            node = prod.get(node.input[0])
            continue
        if t in ("Round", "Sub", "Mul", "Cast"):
            nxt = [i for i in node.input if not is_const(i) and i in prod]
            node = prod.get(nxt[0]) if nxt else None
            continue
        node = None

    if src is None or scale is None:
        return None
    W = _const_value(src, prod, initmap)
    if W is None:
        return None
    return dict(
        src=src,
        W=np.asarray(W, np.float64),
        s=np.asarray(scale, np.float64),
        zp=0.0 if zp is None else float(np.asarray(zp).flatten()[0]),
        lo=-128.0 if lo is None else lo,
        hi=127.0 if hi is None else hi,
        chain_nodes=chain_nodes,
    )


def _quantize(info: dict, is_bias: bool) -> np.ndarray:
    s = info["s"]
    W = info["W"]
    if s.size > 1:
        s = s.reshape([-1] + [1] * (W.ndim - 1)) if not is_bias else s.reshape([-1])
    q = np.round(W / s + info["zp"])
    if not is_bias:
        q = np.clip(q, info["lo"], info["hi"]).astype(np.int8)
    else:
        q = np.clip(q, info["lo"], info["hi"]).astype(np.int32)
    return q


def integerize_perchannel_weights(
    model: onnx.ModelProto,
) -> Tuple[onnx.ModelProto, Dict[str, dict]]:
    """Rewire every Conv/Gemm to read an int8 weight / int32 bias initializer directly,
    deleting the per-channel weight/bias QCDQ+dequant chains. Returns (model, scale_map)
    where scale_map[conv_name] = {"weight_scale": s_w[C], "bias_scale": s_b[C],
    "weight_src": name, "bias_src": name}. The caller uses s_w to build the per-channel
    post-conv RequantShift (mul[c] = s_in·s_w[c]/s_out).
    """
    graph = model.graph
    prod, initmap = _build_maps(graph)
    scale_map: Dict[str, dict] = {}
    dead_nodes: set = set()
    new_inits: List[onnx.TensorProto] = []
    rewire: Dict[str, str] = {}  # consumer_node_name+idx handled via direct input edit

    for n in graph.node:
        if n.op_type not in ("Conv", "Gemm"):
            continue
        entry: dict = {}
        for idx, key, is_bias in ((1, "weight", False), (2, "bias", True)):
            if len(n.input) <= idx:
                continue
            info = trace_weight_qcdq(n.input[idx], prod, initmap)
            if info is None:
                continue
            q = _quantize(info, is_bias)
            new_name = f"{info['src']}_int{'32' if is_bias else '8'}"
            new_inits.append(numpy_helper.from_array(q, name=new_name))
            n.input[idx] = new_name  # consumer now reads the int initializer directly
            dead_nodes.update(info["chain_nodes"])
            entry[f"{key}_scale"] = info["s"]
            entry[f"{key}_src"] = info["src"]
        if entry:
            scale_map[n.name] = entry

    # Remove dead QCDQ/dequant nodes (private single-consumer chains) + their Constants.
    kept = [nd for nd in graph.node if nd.name not in dead_nodes]
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(new_inits)
    return model, scale_map
