# SPDX-License-Identifier: MIT
"""
Pooled-percentile PTQ activation calibration for Brevitas models — the "proper" collector.

Why this exists (verified against the installed Brevitas 0.13.0 sources):

  * Our production calibration (`onnx4deeploy/core/base_exporter.py:706`) is
        with torch.no_grad(), calibration_mode(model): model(calib)
    `calibration_mode` (brevitas/graph/calibrate.py:463-485) disables act/weight/bias
    quantization, puts the quant PROXIES in training mode (calibrate.py:151-179 —
    `module.train(is_training)` is applied to proxy modules only, so fp32 BatchNorm
    keeps whatever mode the model was in), sets `observer_only=True` on the tensor-quant
    (calibrate.py:174-176) so the quantizer impl is CALLED but its output DISCARDED,
    extends `collect_stats_steps` to maxsize and sets `momentum=None`
    (calibrate.py:479-480 -> :77-88).

  * The scale collector itself, `ParameterFromRuntimeStatsScaling.training_forward`
    (brevitas/core/scaling/standalone.py:456-475): per CALL it computes
    `stats = AbsPercentile(this_batch)` and folds it into a buffer with
    `inplace_momentum_update(buffer, stats, momentum, counter, new_counter)` —
    a true running MEAN of per-batch percentiles when momentum is None (inside
    calibration_mode), an EMA (momentum 0.1) otherwise.  Either way the estimate is an
    AVERAGE OF PER-BATCH PERCENTILES, never a quantile of the pooled data.  With one
    calibration call (our production path: a single batch of 8) it is simply the
    99.999-percentile of that one batch — i.e. its max, since 8 windows have far fewer
    than 100k values per site.

  * Scale convention (brevitas/core/quant/int.py:92-135, `RescalingIntQuant`):
    `scaling_impl(x)` returns the float THRESHOLD t; the actual scale is
    `scale = t / int_scaling_impl(bit_width)` (int.py:135).  For Int8ActPerTensorFloat
    (signed, non-narrow) `int_scaling_impl(8) = 128`, which is why the uncalibrated
    default threshold 1.0 shows up as scale 0.0078125 = 1/128.

This module implements the pooled alternative:
    pool |x| at every activation-quantizer site over the WHOLE calibration set
    (streamed in batches, capped with uniform per-batch subsampling), then take ONE
    quantile per site at the end and freeze it as a constant threshold.
Observation happens under `calibration_mode(model)` itself, so the observed tensors are
EXACTLY the ones Brevitas' own collector sees (float network, quant disabled, same
activation_impl); we merely add forward-pre-hooks on each
`proxy.fused_activation_quant_proxy.tensor_quant` (the tensor entering the collector,
runtime_quant.py:81-91: activation_impl is applied before tensor_quant).
Side effect: Brevitas' own buffers also collect during our pass; this is irrelevant
because `freeze_act_thresholds` then force-sets every scale.
"""
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------------------
class ConstThreshold(torch.nn.Module):
    """Replaces a tensor-quant `scaling_impl`; holds a constant float THRESHOLD t.

    Call contract (brevitas/core/quant/int.py:156-160, `RescalingIntQuant.forward`):
        int_threshold = self.int_scaling_impl(bit_width)      # 128 for Int8ActPerTensorFloat
        scale = self.scaling_impl(x, int_threshold)           # impl divides internally
    i.e. the scaling impl RECEIVES the integer threshold and must return t/int_threshold —
    exactly what `ParameterFromRuntimeStatsScaling.forward` does with its buffer/value
    (standalone.py:485-503).  Same surgery recipe as the weight-scale freeze validated in
    exp_brevitas_stall (whose ConstScale stores the already-divided SCALE and ignores the
    threshold argument — for weights the stored quantity was m.quant_weight().scale).
    """

    def __init__(self, v: float):
        super().__init__()
        self.register_buffer("v", torch.tensor(float(v), dtype=torch.float32))

    def forward(self, x=None, threshold=None, *a, **k):
        if threshold is None:
            return self.v
        return self.v / threshold


def act_sites(model) -> Dict[str, torch.nn.Module]:
    """Ordered {site_name: ActQuantProxy} for every ENABLED activation quantizer."""
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, ActQuantProxyFromInjectorBase) \
                and getattr(m, "fused_activation_quant_proxy", None) is not None:
            out[name] = m
    return out


def int_scaling_of(proxy) -> float:
    """int_scaling_impl(bit_width) of a site — 128 for Int8ActPerTensorFloat."""
    tq = proxy.fused_activation_quant_proxy.tensor_quant
    with torch.no_grad():
        bw = tq.msb_clamp_bit_width_impl()
        return float(tq.int_scaling_impl(bw))


# ---------------------------------------------------------------------------------------
class PooledAbsCollector:
    """Pools |x| per site across batches. cap: max pooled values per site (float16 store,
    uniform per-batch subsample when the projected total exceeds the cap); the exact
    global max and the exact total count are always tracked in float32/int."""

    def __init__(self, sites: Dict[str, torch.nn.Module], n_batches: int,
                 cap: int = 40_000_000, seed: int = 0):
        self.sites = sites
        self.n_batches = n_batches
        self.cap = cap
        self.rng = np.random.RandomState(seed)
        self.pool: Dict[str, List[np.ndarray]] = {n: [] for n in sites}
        self.true_max = {n: 0.0 for n in sites}
        self.true_n = {n: 0 for n in sites}
        self.per_batch_keep = max(1, cap // n_batches)
        self._handles = []

    def _hook(self, name):
        def fn(module, inp):
            x = inp[0]
            a = x.detach().abs().reshape(-1)
            self.true_max[name] = max(self.true_max[name], float(a.max()))
            self.true_n[name] += a.numel()
            a = a.cpu().numpy()
            if a.size > self.per_batch_keep:
                a = a[self.rng.choice(a.size, self.per_batch_keep, replace=False)]
            self.pool[name].append(a.astype(np.float16))
        return fn

    def __enter__(self):
        for n, p in self.sites.items():
            tq = p.fused_activation_quant_proxy.tensor_quant
            self._handles.append(tq.register_forward_pre_hook(self._hook(n)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []

    def finalize(self) -> Dict[str, dict]:
        """{site: {n_total, n_pooled, kept_fraction, max}} + pooled arrays retained."""
        out = {}
        for n in self.sites:
            arr = np.concatenate(self.pool[n]) if self.pool[n] else np.zeros(0, np.float16)
            self.pool[n] = [arr]
            out[n] = dict(n_total=int(self.true_n[n]), n_pooled=int(arr.size),
                          kept_fraction=float(arr.size / max(1, self.true_n[n])),
                          abs_max=float(self.true_max[n]))
        return out

    def quantile(self, name: str, pct: float) -> float:
        """Threshold for a site at `pct` (100 -> the EXACT tracked max)."""
        if pct >= 100.0:
            return float(self.true_max[name])
        arr = self.pool[name][0].astype(np.float32)
        return float(np.percentile(arr, pct))


# ---------------------------------------------------------------------------------------
@torch.no_grad()
def collect_pooled(model, X: np.ndarray, batch_size: int = 64,
                   cap: int = 40_000_000, seed: int = 0, log=print):
    """Run the pooled collection pass. `model` must be in eval() (BN frozen); observation
    happens inside brevitas' own calibration_mode so conditions are identical to the
    production collector (float activations, quant disabled, proxies in observer mode).

    Returns (collector, site_meta).
    """
    from brevitas.graph.calibrate import calibration_mode
    assert not model.training, "call model.eval() first — BN must stay on running stats"
    sites = act_sites(model)
    nb = int(np.ceil(X.shape[0] / batch_size))
    log(f"[collect] {X.shape[0]} windows, batch {batch_size} -> {nb} collector steps, "
        f"{len(sites)} act sites, cap {cap/1e6:.0f}M/site")
    coll = PooledAbsCollector(sites, nb, cap=cap, seed=seed)
    with coll, calibration_mode(model):
        for i in range(0, X.shape[0], batch_size):
            model(torch.from_numpy(X[i:i + batch_size]))
    model.eval()
    meta = coll.finalize()
    for n, m in meta.items():
        log(f"   {n:45s} N={m['n_total']:>11,d}  pooled={m['n_pooled']:>10,d} "
            f"({100*m['kept_fraction']:5.1f}%)  max={m['abs_max']:.4f}")
    return coll, meta


def thresholds_from_pool(coll: PooledAbsCollector, percentiles) -> Dict[float, Dict[str, dict]]:
    """{pct: {site: {threshold, rank}}} — rank = expected #values above the threshold."""
    out = {}
    for pct in percentiles:
        d = {}
        for n in coll.sites:
            t = coll.quantile(n, pct)
            n_tot = coll.true_n[n]
            d[n] = dict(threshold=t, pct=pct,
                        rank_from_top=float(n_tot * (1.0 - min(pct, 100.0) / 100.0)))
        out[pct] = d
    return out


def freeze_act_thresholds(model, thresholds: Dict[str, float], log=print) -> Dict[str, float]:
    """ConstThreshold surgery on every act site; verifies proxy.scale() == t/int_scaling
    bit-exactly. Returns {site: frozen_scale}."""
    sites = act_sites(model)
    missing = set(thresholds) - set(sites)
    assert not missing, f"unknown sites: {missing}"
    frozen = {}
    for n, p in sites.items():
        t = float(thresholds[n])
        isc = int_scaling_of(p)
        p.fused_activation_quant_proxy.tensor_quant.scaling_impl = ConstThreshold(t)
        with torch.no_grad():
            got = float(p.scale())
        want = t / isc
        assert abs(got - want) <= 1e-7 * max(abs(want), 1e-30), \
            f"{n}: frozen scale {got!r} != {want!r} (t={t}, int_scaling={isc})"
        frozen[n] = got
    return frozen


@torch.no_grad()
def read_act_scales(model) -> Dict[str, float]:
    return {n: float(p.scale()) for n, p in act_sites(model).items()}
