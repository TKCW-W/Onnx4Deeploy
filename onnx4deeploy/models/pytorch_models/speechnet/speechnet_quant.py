# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
QuantSpeechNet — INT8 Brevitas-quantized SpeechNet for the quantized-ZO (QZO) datapath.

Design (feat/QZO, QZO_exp/exp1) — clean build modeled on the shipped QMCUNet-In1 quant pattern:
  * Conv weights: INT8 per-channel  (Int8WeightPerChannelFloat)
  * Conv bias:    INT32             (Int32Bias)
  * activations:  INT8 per-tensor   (Int8ActPerTensorFloat), including a **conv-output** quantizer
                  (output_quant) so every conv has a real output scale s_c  (the B1 fix)
  * BatchNorm:    **unfolded, plain nn.BatchNorm2d in fp32** (γ/β trainable, trained in the fp32 path)
  * FC (Gemm):    weight+bias quantized INT8 (Int8WeightPerChannelFloat / Int32Bias); logits fp32 for the loss

Datapath Brevitas emits per block:  int8-Conv → (dequant) → fp32 BN → ReLU → MaxPool → (re-quant) → int8-Conv.
Only Conv (w,b) and FC (w,b) are quantized; BN stays fp32. Architecture matches SpeechNetDeploy exactly.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Int32Bias

# (out_channels, kernel, pool) per block — identical to SpeechNetDeploy's default blocks_config.
_BLOCKS = [
    (8, (1, 4), (1, 8)),
    (16, (1, 16), (1, 4)),
    (16, (1, 8), (1, 4)),
    (32, (7, 1), (1, 1)),
    (32, (7, 1), (1, 1)),
]


def _conv_kwargs(conv_output_quant: bool) -> Dict[str, Any]:
    """Int8 QuantConv2d config. output_quant present => conv-output observer (B1 fix)."""
    kw = dict(
        weight_bit_width=8,
        weight_quant=Int8WeightPerChannelFloat,   # per-channel int8 weights
        bias_quant=Int32Bias,                      # int32 bias
        input_quant=Int8ActPerTensorFloat,         # int8 activation in
        return_quant_tensor=True,
    )
    if conv_output_quant:
        kw["output_quant"] = Int8ActPerTensorFloat  # int8 conv output => real s_c (post-conv, pre-BN)
    return kw


class _QuantBlock(nn.Module):
    """int8 QuantConv2d -> fp32 BN (unfolded) -> ReLU -> (MaxPool | Identity)."""

    def __init__(self, in_ch: int, out_ch: int, kernel, pool, conv_output_quant: bool = True):
        super().__init__()
        k_c, k_t = int(kernel[0]), int(kernel[1])
        self.conv = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=(k_c, k_t), stride=(1, 1),
            padding=(0, k_t // 2), bias=True, **_conv_kwargs(conv_output_quant),
        )
        self.bn = nn.BatchNorm2d(out_ch)            # <-- fp32, unfolded, trainable γ/β
        self.relu = nn.ReLU(inplace=False)
        pc, pt = int(pool[0]), int(pool[1])
        self.pool: nn.Module = nn.Identity() if (pc == 1 and pt == 1) \
            else nn.MaxPool2d(kernel_size=(pc, pt), stride=(pc, pt))

    def forward(self, x):
        # conv returns an int8 QuantTensor; BN (plain fp32) unwraps it to float -> fp32 BN/ReLU/MaxPool.
        return self.pool(self.relu(self.bn(self.conv(x))))


class QuantSpeechNetDeploy(nn.Module):
    """INT8 Brevitas SpeechNet (conv+fc quantized, BN unfolded fp32)."""

    def __init__(self, num_channels: int = 14, time_steps: int = 700, num_classes: int = 9,
                 conv_output_quant: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.num_classes = num_classes

        # NOTE: no standalone input QuantIdentity — block 0's QuantConv2d.input_quant already quantises the
        # raw window, so the input is quantised exactly ONCE (avoids the redundant double input-quant).
        self.blocks = nn.ModuleList()
        in_ch = 1
        for out_ch, kernel, pool in _BLOCKS:
            self.blocks.append(_QuantBlock(in_ch, out_ch, kernel, pool, conv_output_quant=conv_output_quant))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc_in = in_ch
        # GAP output -> int8 before the quantised FC.
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc = qnn.QuantLinear(
            in_ch, num_classes, bias=True,
            weight_bit_width=8, weight_quant=Int8WeightPerChannelFloat, bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat, return_quant_tensor=False,  # fp32 logits for the loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:            # block 0's conv.input_quant quantises the raw input
            x = block(x)
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], self._fc_in)
        x = self.fc_iq(x)
        return self.fc(x)
