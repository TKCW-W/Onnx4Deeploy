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
from onnx import numpy_helper, TensorProto

TP_I8, TP_I32, TP_F32, TP_I64 = TensorProto.INT8, TensorProto.INT32, TensorProto.FLOAT, TensorProto.INT64


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


def _prune_orphans(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove nodes whose outputs are consumed by nothing (dead Constants/Casts left by the QCDQ
    removal) + orphan initializers. Iterate to a fixpoint. Keeps the graph clean for Deeploy."""
    g = model.graph
    graph_outs = {o.name for o in g.output}
    changed = True
    while changed:
        used = set(graph_outs)
        for n in g.node:
            used.update(n.input)
        keep = [n for n in g.node if any(o in used for o in n.output) or not n.output]
        changed = len(keep) != len(g.node)
        del g.node[:]; g.node.extend(keep)
    used_inits = set()
    for n in g.node:
        used_inits.update(n.input)
    keep_i = [i for i in g.initializer if i.name in used_inits]
    del g.initializer[:]; g.initializer.extend(keep_i)
    return model


def _annotate_shapes_via_run(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give every intermediate tensor an exact shape/dtype by running the graph once (the custom
    Quant/Dequant/RequantShift ops block ONNX shape inference; Deeploy's _assertTensorsHaveShape needs
    them all). Requests every node output from run_onnx_graph on a zero int8 input; weights are still
    initializers here so only `input` must be fed."""
    import tempfile as _tf, os as _os
    from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
    _np_dtype = {np.dtype("int8"): TP_I8, np.dtype("int32"): TP_I32, np.dtype("float32"): TP_F32,
                 np.dtype("int64"): TP_I64}
    g = model.graph
    shape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    # dummy matches the graph input's dtype (int8 for offline-quantised input, fp32 for online-quantised). -- QW
    _np_of = {TP_I8: np.int8, TP_I32: np.int32, TP_F32: np.float32, TP_I64: np.int64}
    dummy = np.zeros(shape, dtype=_np_of.get(g.input[0].type.tensor_type.elem_type, np.float32))
    have = {o.name for o in g.output} | {i.name for i in g.input} | {i.name for i in g.initializer}
    outs = [o for n in g.node for o in n.output if o not in have]
    tmp = _os.path.join(_tf.gettempdir(), "_qzo_shape.onnx")
    onnx.save(model, tmp)
    try:
        res = run_onnx_graph(tmp, {g.input[0].name: dummy}, output_names=outs)
    finally:
        try: _os.remove(tmp)
        except OSError: pass
    captured = {name: np.asarray(arr) for name, arr in zip(outs, res)}
    # drop any stale (possibly shapeless) value_info for captured names, then re-add exact shapes.
    keep = [v for v in g.value_info if v.name not in captured]
    del g.value_info[:]
    g.value_info.extend(keep)
    for name, a in captured.items():
        g.value_info.append(onnx.helper.make_tensor_value_info(
            name, _np_dtype.get(a.dtype, TP_F32), list(a.shape)))
    return model


def _toposort(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    have = {i.name for i in g.initializer} | {i.name for i in g.input}
    nodes, out, changed = list(g.node), [], True
    while nodes and changed:
        changed, rest = False, []
        for n in nodes:
            if all(inp == "" or inp in have for inp in n.input):
                out.append(n); have.update(n.output); changed = True
            else:
                rest.append(n)
        nodes = rest
    out.extend(nodes)
    del g.node[:]; g.node.extend(out)
    return model


def build_int8_forward(
    model: onnx.ModelProto,
    div: int = 1 << 16,
    s_in: float = 1.0 / 128,
    s_out: float = 1.0 / 128,
) -> Tuple[onnx.ModelProto, Dict[str, dict]]:
    """Turn the `-mode quant` QCDQ graph into a NUMERICALLY-CORRECT int8 datapath.

    The pipeline's own `network.onnx` is a broken hybrid (fp32 conv + a scalar RequantShift
    whose ``mul`` omits ``s_in`` and per-channel ``s_w`` — ~128x off, outputs zeros). This
    rebuilds the correct integer datapath from the per-channel scales:

      conv:  a_int8 → Conv(int8 weight, int32 bias) → RequantShift(mul[c]=round(s_w[c]·div),
             add=0, div) → Dequant(s_out) → Relu → MaxPool → Quant(s_in_next) → …
      fc  :  … → Gemm(int8 weight, int32 bias) → per-channel dequant Mul(s_in·s_w[c]) → logits

    Bias stays as the Conv/Gemm 3rd input (the shipped q-zo convention → both weight and bias
    are directly perturbable graph inputs for ZO). Assumes uniform activation scale
    ``s_in == s_out`` (true for SpeechNet: all 1/128), so ``mul[c] = round(s_w[c]·div)``.

    VERIFIED: eps-0 forward matches the PyTorch/Brevitas reference (`outputs.npz`) —
    cos≈1.0000, maxdiff≈0.03, argmax match (div=2**16).

    Returns (model, scale_map). scale_map[conv] carries weight_scale/bias_scale for the ZO
    perturbation-magnitude (RQSPerturb mul) downstream.
    """
    model, scale_map = integerize_perchannel_weights(model)
    g = model.graph
    initmap = {i.name: i for i in g.initializer}
    cons: Dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    graph_outs = {o.name for o in g.output}

    for n in list(g.node):
        if n.op_type not in ("Conv", "Gemm"):
            continue
        ent = scale_map.get(n.name)
        if ent is None:
            continue
        s_w = np.asarray(ent["weight_scale"], np.float64).reshape(-1)
        # QW: per-layer activation scales (2026-09-04). The uniform s_in==s_out assumption held
        #     only for the OLD uncalibrated 1/128 scales; under the pooled@99.99 calibration the
        #     per-tensor scales differ per layer (e.g. block0 in=22.297, out=13.1875 → ratio
        #     1.69), and dropping the ratio distorts every block's output by s_in/s_out
        #     (found via the Brevitas cross-check in exp9's build_qzo_infer_fixture: logit
        #     cos 0.87 instead of ≈1). Trace THIS conv's input Quant scale and output Dequant
        #     scale from the graph; fall back to the function args only if not found. -- QW
        def _scale_attr(_nd):
            for _a in _nd.attribute:
                if _a.name == "scale":
                    return float(onnx.helper.get_attribute_value(_a))
            return None

        _prod = {o: pn for pn in g.node for o in pn.output}
        s_in_l, cur = None, n.input[0]
        for _ in range(6):                       # walk up through pass-through ops to the Quant
            pn = _prod.get(cur)
            if pn is None:
                break
            if pn.op_type == "Quant":
                s_in_l = _scale_attr(pn); break
            if not pn.input:
                break
            cur = pn.input[0]
        rqs = next((c for c in cons.get(n.output[0], []) if c.op_type == "RequantShift"), None)
        s_out_l = None
        if rqs is not None:
            cur = rqs.output[0]
            for _ in range(6):                   # walk down to the Dequant
                dn = next((c for c in cons.get(cur, [])), None)
                if dn is None:
                    break
                if dn.op_type == "Dequant":
                    s_out_l = _scale_attr(dn); break
                if not dn.output:
                    break
                cur = dn.output[0]
        if rqs is not None and (s_in_l is None or s_out_l is None):
            # QW: only meaningful on the requantized (conv) path; the float-fc Gemm has no
            #     RequantShift/Dequant to trace and never uses these values. -- QW
            print(f"   ⚠ {n.name}: per-layer act scale not traced "
                  f"(s_in={s_in_l}, s_out={s_out_l}) — falling back to uniform args")
        _si = s_in_l if s_in_l is not None else s_in
        _so = s_out_l if s_out_l is not None else s_out
        if rqs is not None:
            # conv: per-channel RequantShift mul; the int32 bias moves from the Conv into the RequantShift
            # `add`, so the Conv is 2-input (data_in, weight). This matches EVERY shipped PULP int8 conv
            # (all `*_RQ` tests are 2-input, bias-in-add) and the int8 pulp_nn_conv kernel, which has no bias
            # arg — the Conv+RequantShift merge then yields the canonical 4-input RequantizedConv
            # (data_in, weight, mul, add). The `add` is a per-channel int32 variable → still ZO-perturbable.
            # add[c] = round(s_w[c]·bias_int32[c]·div) = round(bias_fp32[c]·div/s_in); with s_in==s_out (all
            # 1/128 for SpeechNet) this is the requant-add-domain bias round(bias_fp32·div/s_out). -- QW
            mul = np.round(s_w * (_si / _so) * div).astype(np.int32)   # QW: per-layer s_in/s_out
            b_name = n.input[2] if len(n.input) > 2 else None
            if b_name is not None and b_name in initmap:
                b = numpy_helper.to_array(initmap[b_name]).astype(np.float64).reshape(-1)
                # QW: same per-layer ratio for the add — exact form add = b_int·mul_exact,
                #     i.e. round(s_w·b_int·(s_in/s_out)·div) = round(b_fp32/s_out·div). -- QW
                add = np.round(s_w * b * (_si / _so) * div).astype(np.int32)  # int32 bias in RQS-add units
                add_name = f"{ent['bias_src']}_rqsadd"
                del n.input[2]                                         # Conv now 2-input (weight only)
                ent["bias_rqs_name"] = add_name                       # perturbable int32 bias lives here
                ent["bias_rqs_node"] = rqs.name
            else:
                add = np.zeros_like(mul); add_name = rqs.input[2] + "_pc"
            nm = numpy_helper.from_array(mul, name=rqs.input[1] + "_pc")
            na = numpy_helper.from_array(add, name=add_name)
            rqs.input[1], rqs.input[2] = nm.name, na.name
            g.initializer.extend([nm, na])
            for a in list(rqs.attribute):
                if a.name == "div":
                    rqs.attribute.remove(a)
            rqs.attribute.append(
                onnx.helper.make_attribute("div", numpy_helper.from_array(np.array(div, np.int64)))
            )
        elif n.output[0] in graph_outs:
            # FLOAT fc (mirror the shipped QMCUNetZO ZO fixture: quantize the Conv layers only, keep the
            # classifier Gemm in fp32). Dequantize the fc weight/bias back to plain fp32 initializers
            # (perturbed later by float PerturbRademacher, exactly like BN γ/β) and bypass the fc input
            # activation quant so the Gemm consumes the fp32 GAP/Reshape features → a clean 3-input float
            # Gemm (data, weight, bias) → fp32 logits. int8 conv datapath + fp32 head = QMCUNetZO. -- QW
            w_int8 = numpy_helper.to_array(initmap[n.input[1]]).astype(np.float64)
            w_fp32 = (w_int8 * s_w.reshape([-1] + [1] * (w_int8.ndim - 1))).astype(np.float32)
            wname = f"{ent['weight_src']}_fp32"
            g.initializer.append(numpy_helper.from_array(w_fp32, name=wname))
            n.input[1] = wname                                        # fp32 weight (float-perturbable)
            if len(n.input) > 2 and n.input[2] in initmap:
                s_b = np.asarray(ent["bias_scale"], np.float64).reshape(-1)
                b_fp32 = (numpy_helper.to_array(initmap[n.input[2]]).astype(np.float64).reshape(-1) * s_b).astype(np.float32)
                bname = f"{ent['bias_src']}_fp32"
                g.initializer.append(numpy_helper.from_array(b_fp32, name=bname))
                n.input[2] = bname                                    # fp32 bias stays the Gemm's 3rd input
            # bypass the fc input activation quant: walk back through the Quant/RequantShift chain to the
            # first fp32 producer (Reshape/GAP) and feed that directly; the orphaned quant nodes get pruned.
            _q = {"Quant", "Dequant", "RequantShift", "Clip", "Div", "Add", "Round", "Sub", "Mul"}
            src, seen = n.input[0], set()
            while src not in seen:
                seen.add(src)
                pr = next((p for p in g.node if src in p.output), None)
                if pr is None or pr.op_type not in _q:
                    break
                src = pr.input[0]
            n.input[0] = src                                          # fp32 features → float Gemm
            # n.output[0] is already the graph output → Gemm emits fp32 logits directly (no Mul/Add).

    # Re-emit each unfolded fp32 BatchNormalization as BatchNormInternal (com.microsoft, training_mode=1,
    # 5 outputs) — the ORT training-mode BN, matching the float-ZO exp6 fixture. γ/β stay as inputs (promoted
    # + perturbed downstream); running_mean/var stay frozen initializers. -- QW
    initset = {i.name for i in g.initializer}
    new_nodes = []
    for n in g.node:
        if n.op_type == "BatchNormalization":
            eps_a = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
            mom_a = next((a.f for a in n.attribute if a.name == "momentum"), 0.9)
            extra = [f"{n.name}_rm", f"{n.name}_rv", f"{n.name}_sm", f"{n.name}_siv"]
            bn = onnx.helper.make_node("BatchNormInternal", list(n.input), [n.output[0]] + extra,
                                       name=n.name, domain="com.microsoft",
                                       epsilon=float(eps_a), momentum=float(mom_a), training_mode=1)
            new_nodes.append(bn)
            gi = initmap.get(n.input[1])
            C = int(gi.dims[0]) if gi is not None else 0
            for eo in extra:
                g.value_info.append(onnx.helper.make_tensor_value_info(eo, TP_F32, [C]))
        else:
            new_nodes.append(n)
    del g.node[:]; g.node.extend(new_nodes)

    for n in g.node:
        if n.op_type in ("Quant", "Dequant", "RequantShift"):
            n.domain = "ai.onnx.contrib"
    model = _toposort(_prune_orphans(model))
    _annotate_shapes_via_run(model)          # exact intermediate shapes for Deeploy's shape assert -- QW
    return model, scale_map
