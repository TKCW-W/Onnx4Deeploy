# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Quantization scale dump (QZO).

Extracts the per-channel weight scales (and per-tensor input/output activation scales, and int32 bias scales)
from a *calibrated* Brevitas model and writes them to the flat JSON format that
``onnx4deeploy.transform.zo_transform._get_weight_scale`` consumes:

    { "<layer>.weight_quant": [per-channel …],   # int8 weight scale, one per output channel
      "<layer>.bias_quant":   [per-channel …],   # int32 bias scale
      "<layer>.input_quant":  <float>,           # per-tensor input activation scale
      "<layer>.output_quant": <float> }          # per-tensor conv-output activation scale (s_c)

This is the single source of truth for the scale used by BOTH the online weight ``Quant`` node and the
``RQSPerturbRademacher`` ``mul`` (= round(eps/scale * 2**15)), so they stay consistent (QZO prereq #2).
There is no shipped producer for this file — this is it.
"""
import json
from typing import Dict

import numpy as np
import torch


def _scale_of(qtensor) -> np.ndarray:
    if qtensor is None or qtensor.scale is None:
        return None
    return qtensor.scale.detach().cpu().float().numpy().flatten()


def dump_brevitas_scales(model: torch.nn.Module, path: str) -> Dict[str, list]:
    """Walk the (calibrated, eval) Brevitas model and dump every conv/linear layer's scales to `path`."""
    import brevitas.nn as qnn

    model.eval()
    scales: Dict[str, list] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            is_conv = isinstance(mod, qnn.QuantConv2d)
            is_lin = isinstance(mod, qnn.QuantLinear)
            if not (is_conv or is_lin):
                continue
            # weight scale (per output channel)
            try:
                ws = _scale_of(mod.quant_weight())
                if ws is not None:
                    scales[f"{name}.weight_quant"] = ws.tolist()
            except Exception as e:
                print(f"  [scale-dump] {name}: weight scale unavailable ({e})")
            # bias scale (int32) — scale = input_scale * weight_scale
            try:
                if getattr(mod, "bias", None) is not None:
                    bs = _scale_of(mod.quant_bias())
                    if bs is not None:
                        scales[f"{name}.bias_quant"] = bs.tolist()
            except Exception:
                pass
            # input / output activation scales (per tensor)
            try:
                if mod.input_quant is not None and mod.input_quant.is_quant_enabled:
                    s = _scale_of(mod.input_quant(torch.zeros(1, *([1] * 3)) if is_conv else torch.zeros(1, 1)))
            except Exception:
                pass
            for tag in ("input_quant", "output_quant"):
                q = getattr(mod, tag, None)
                try:
                    if q is not None and getattr(q, "is_quant_enabled", False):
                        sc = q.scale() if callable(getattr(q, "scale", None)) else None
                        if sc is not None:
                            scales[f"{name}.{tag}"] = float(np.asarray(sc.detach().cpu()).flatten()[0])
                except Exception:
                    pass

    with open(path, "w") as f:
        json.dump(scales, f, indent=2)
    n_w = sum(1 for k in scales if k.endswith(".weight_quant"))
    print(f"  [scale-dump] wrote {len(scales)} entries ({n_w} weight_quant) -> {path}")
    return scales
