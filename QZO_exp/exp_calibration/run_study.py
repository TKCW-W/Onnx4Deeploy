# SPDX-License-Identifier: MIT
"""
exp_calibration — sweep + metrics + stall linkage (see Plan.md, the contract).

Stages (env STAGE=all|data|pools|sweep|probe|regimes|plots, resumable — every stage
checks results.json / cache files and skips what is already done):

  data    : load & cache the datasets (pretrain 1800 = S01/vocalized sess 1+2 x batch 1..5;
            FT = 54 stratified windows sess3/batch1 seed 42; eval = WHOLE sess3/batch2, 180)
  pools   : pooled |x| collection on the 1800 pretrain windows (29 batches of 64) ->
            thresholds @ {99.9, 99.99, 99.999, 100}; plus batch-2 |x| subsamples for the
            saturation/zero-bin rates
  sweep   : per config (pooled@pct x4, old8, old54, shipped) freeze scales, measure
            zero-shot bal-acc / logit cosine vs float / mean|dCE| / per-site sat+zero rates
  probe   : 100 ZO probe steps at theta0 under best-pooled and old54 (shared z), g stats,
            implied per-channel update in LSB, 0.5-LSB clearance fraction
  regimes : 300-step ZO fine-tune x {direct,master} x {best-pooled, old54}, lr 3e-6,
            shared z/window schedule, eval every 50 steps
  plots   : all figures

Regime/probe machinery adapted from ../exp_masterweight_ft/masterweight_ft.py (cannot be
imported: importing it re-opens/truncates that experiment's run.log at module level).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from calib_pooled import (act_sites, collect_pooled, freeze_act_thresholds,
                          int_scaling_of, read_act_scales, thresholds_from_pool,
                          PooledAbsCollector, ConstThreshold)

CKPT = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
        "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
DATA_PATH = "/app/SilentWear/SilentWear_data/data_raw_and_filt"

EPS = 0.01
N_ACCUM = 4
SEED = 42
N_TRAIN = 54
LR_FT = 3e-6                       # best master lr from exp_masterweight_ft results.json
PCTS = [99.9, 99.99, 99.999, 100.0]
N_PROBE = int(os.environ.get("N_PROBE", 100))
N_REG = int(os.environ.get("N_REG_STEPS", 300))
REG_EVAL_EVERY = 50
CALIB_BATCH = 64                   # 1800/64 -> 29 collector steps (28 full + one of 8)

RESULTS = HERE / "results.json"
DATA_CACHE = HERE / "data_cache.npz"
POOLS_CACHE = HERE / "pools_cache.npz"

_LOG = open(HERE / "run.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


def load_results():
    if RESULTS.exists():
        return json.load(open(RESULTS))
    return {}


def save_results(r):
    tmp = RESULTS.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(r, f, indent=2)
    tmp.replace(RESULTS)


# ---------------------------------------------------------------------------------------
# model / data plumbing (same construction as exp_masterweight_ft)
# ---------------------------------------------------------------------------------------
def make_exporter(session: int, batch: int, calib_samples: int = 8):
    from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter
    e = SpeechNetExporter()
    e._config_overrides = dict(
        dataset="silentwear", data_path=DATA_PATH, pretrained_weights=CKPT,
        pretrained_key="model_state_dict", subject="S01", session=session, batch=batch,
        condition="vocalized", batch_size=1, num_classes=9,
        calib_samples=calib_samples, stratified_sampling=True)
    e.config = e.load_config()
    return e


def stage_data():
    if DATA_CACHE.exists():
        log("[data] cache exists, skipping")
        return
    t0 = time.time()
    pre = []
    for sess in (1, 2):
        for b in range(1, 6):
            e = make_exporter(sess, b)
            xs, _ = e.get_data_source()._load_windows()
            x = np.concatenate(xs, 0).astype(np.float32)
            assert x.shape[0] == 180, f"sess{sess} batch{b}: {x.shape[0]} windows != 180"
            pre.append(x)
            log(f"[data] pretrain sess{sess} batch{b}: {x.shape}")
    preX = np.concatenate(pre, 0)
    assert preX.shape[0] == 1800, f"pretrain total {preX.shape[0]} != 1800"

    e1 = make_exporter(3, 1)
    ishape = e1.get_input_shape()
    X, Y = e1.get_data_source().load_batches(N_TRAIN, (1,) + tuple(ishape[1:]), 9, seed=SEED)
    trX = np.concatenate([np.asarray(a, np.float32) for a in X], 0)
    trY = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)

    e2 = make_exporter(3, 2)
    i2, l2 = e2.get_data_source()._load_windows()
    evX = np.concatenate(i2, 0).astype(np.float32)
    evY = np.concatenate(l2, 0).astype(np.int64).reshape(-1)
    log(f"[data] pretrain {preX.shape}  train {trX.shape}  eval {evX.shape}  "
        f"({time.time()-t0:.0f}s)")
    np.savez_compressed(DATA_CACHE, preX=preX, trX=trX, trY=trY, evX=evX, evY=evY)


def get_data():
    d = np.load(DATA_CACHE)
    return d["preX"], d["trX"], d["trY"], d["evX"], d["evY"]


def build_model():
    """Fresh Brevitas model, pretrained weights, eval BN. NO calibration here — every
    config's act scales are force-set afterwards (calibration only matters for old8/old54,
    which run it on scratch models in derive_old_scales)."""
    e = make_exporter(3, 1)
    m = e.create_brevitas_model()
    m.eval()
    return e, m


def build_float_model():
    e = make_exporter(3, 1)
    fm = e.create_model()
    fm.eval()
    return fm


# ---------------------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------------------
def stage_pools():
    r = load_results()
    if "pools" in r and POOLS_CACHE.exists():
        log("[pools] done, skipping")
        return
    preX, trX, trY, evX, evY = get_data()
    _, model = build_model()

    coll, meta = collect_pooled(model, preX, batch_size=CALIB_BATCH, cap=40_000_000,
                                seed=SEED, log=log)
    th = thresholds_from_pool(coll, PCTS)

    # batch-2 populations for the sat/zero-bin rates (float-net observation, same hooks)
    coll2, meta2 = collect_pooled(model, evX, batch_size=CALIB_BATCH, cap=4_000_000,
                                  seed=SEED + 1, log=log)

    save = {}
    for n in coll2.sites:
        save[f"b2_{n}"] = coll2.pool[n][0]                     # float16 |x| subsample
        save[f"pre_{n}"] = coll.pool[n][0][:2_000_000]         # for the scale plot context
    np.savez_compressed(POOLS_CACHE, **save)

    r["pools"] = dict(
        n_windows=int(preX.shape[0]), batch=CALIB_BATCH,
        n_collector_steps=int(np.ceil(preX.shape[0] / CALIB_BATCH)),
        site_meta=meta,
        thresholds={f"{p:g}": th[p] for p in PCTS},
        batch2_site_meta=meta2,
    )
    save_results(r)
    log(f"[pools] thresholds for {len(meta)} sites at {PCTS} saved")


# ---------------------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------------------
def derive_old_scales(n_windows: int, trX: np.ndarray) -> dict:
    """Reproduce the PRODUCTION calibration on a scratch model: one calibration_mode call
    with the first n_windows FT windows (base_exporter.py:706 style), then read the scales.
    Returns {site: scale}."""
    from brevitas.graph.calibrate import calibration_mode
    _, m = build_model()
    with torch.no_grad(), calibration_mode(m):
        m(torch.from_numpy(trX[:n_windows]))
    m.eval()
    return read_act_scales(m)


def config_thresholds(r, key: str, trX) -> dict:
    """-> {site: threshold}. Also memoises the derived old8/old54 scales in results."""
    sites_meta = r["pools"]["site_meta"]
    if key.startswith("pooled@"):
        pct = key.split("@")[1]
        return {n: v["threshold"] for n, v in r["pools"]["thresholds"][pct].items()}
    if key == "shipped":
        return {n: 1.0 for n in sites_meta}          # default threshold -> scale 1/128
    if key in ("old8", "old54"):
        cache = r.setdefault("old_scales", {})
        if key not in cache:
            n = 8 if key == "old8" else 54
            log(f"[sweep] deriving {key} via production calibration_mode({n} windows)")
            sc = derive_old_scales(n, trX)
            cache[key] = sc
            save_results(r)
        # threshold = scale * int_scaling (int_scaling read from a live model at freeze
        # time; Int8ActPerTensorFloat -> 128 for every site, asserted in freeze)
        return {s: None for s in cache[key]}          # sentinel: freeze from scale
    raise KeyError(key)


def freeze_config(model, r, key: str, trX) -> dict:
    th = config_thresholds(r, key, trX)
    if key in ("old8", "old54"):
        sc = r["old_scales"][key]
        th = {n: sc[n] * int_scaling_of(p) for n, p in act_sites(model).items()}
    frozen = freeze_act_thresholds(model, th, log=log)
    return frozen


# ---------------------------------------------------------------------------------------
# eval helpers (adapted from masterweight_ft.py)
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def logits_of(model, X, chunk=60):
    return np.concatenate([model(torch.from_numpy(X[i:i + chunk])).numpy()
                           for i in range(0, X.shape[0], chunk)], 0)


def balanced(logits, Y):
    pred = logits.argmax(-1)
    rec = {int(c): float((pred[Y == c] == c).mean()) for c in np.unique(Y)}
    return float(np.mean(list(rec.values())) * 100.0), \
        float((pred == Y).mean() * 100.0), rec


def per_window_ce(logits, Y):
    lt = torch.from_numpy(logits)
    yt = torch.from_numpy(Y)
    return F.cross_entropy(lt, yt, reduction="none").numpy()


def site_health(scales: dict, npz) -> dict:
    """sat = P(|x| > 127*s) (clipped), zero = P(|x| < s/2) (rounds to 0) on the batch-2
    float-net populations."""
    out = {}
    for n, s in scales.items():
        a = npz[f"b2_{n}"].astype(np.float32)
        out[n] = dict(scale=float(s),
                      sat_rate=float((a > 127.0 * s).mean()),
                      zero_rate=float((a < 0.5 * s).mean()))
    return out


def stage_sweep():
    r = load_results()
    preX, trX, trY, evX, evY = get_data()
    npz = np.load(POOLS_CACHE)
    sweep = r.setdefault("sweep", {})

    if "float_ref" not in r:
        fm = build_float_model()
        fl = logits_of(fm, evX, chunk=1)     # chunk=1: BN-free model but keep it simple/safe
        bal, ov, rec = balanced(fl, evY)
        r["float_ref"] = dict(balanced_accuracy=bal, overall_accuracy=ov,
                              ce=per_window_ce(fl, evY).tolist())
        np.save(HERE / "float_logits.npy", fl)
        save_results(r)
        log(f"[sweep] float ref bal_acc={bal:.2f}%")
    fl = np.load(HERE / "float_logits.npy")
    fce = np.asarray(r["float_ref"]["ce"], np.float32)

    configs = [f"pooled@{p:g}" for p in PCTS] + ["old8", "old54", "shipped"]
    _, model = build_model()
    for key in configs:
        if key in sweep:
            log(f"[sweep] {key} done, skipping")
            continue
        t0 = time.time()
        frozen = freeze_config(model, r, key, trX)
        ql = logits_of(model, evX)
        bal, ov, rec = balanced(ql, evY)
        qce = per_window_ce(ql, evY)
        cos = float(np.mean(np.sum(ql * fl, -1) /
                            (np.linalg.norm(ql, axis=-1) * np.linalg.norm(fl, axis=-1) + 1e-12)))
        sweep[key] = dict(
            scales=frozen,
            balanced_accuracy=bal, overall_accuracy=ov, per_class_recall=rec,
            logit_cosine_vs_float=cos,
            mean_abs_dCE_vs_float=float(np.abs(qce - fce).mean()),
            mean_CE=float(qce.mean()),
            site_health=site_health(frozen, npz),
        )
        save_results(r)
        log(f"[sweep] {key:14s} bal_acc={bal:6.2f}%  cos={cos:.4f}  "
            f"|dCE|={sweep[key]['mean_abs_dCE_vs_float']:.4f}  ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------------------
# ZO machinery (adapted from masterweight_ft.py — kept behaviourally identical)
# ---------------------------------------------------------------------------------------
class ConstScale(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("v", v.detach().clone())

    def forward(self, *a, **k):
        return self.v


def build_params(model):
    """Freeze weight scales (ConstScale) + collect the 22 trainable tensors. Bias grid
    s_b = s_in * s_w uses the CURRENT (frozen) act input scale — config-dependent."""
    import brevitas.nn as qnn
    params = []
    for name, m in model.named_modules():
        if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear)):
            s_w = m.quant_weight().scale.detach().clone()
            s_in = m.input_quant.scale().detach().clone()
            m.weight_quant.tensor_quant.scaling_impl = ConstScale(s_w)
            params.append(dict(name=f"{name}.weight", mod=m, attr="weight", kind="quant",
                               scale=s_w, lo=-127, hi=127, init=m.weight.detach().clone()))
            s_b = (s_in * s_w).reshape(-1)
            params.append(dict(name=f"{name}.bias", mod=m, attr="bias", kind="quant",
                               scale=s_b, lo=-(2 ** 31) + 1, hi=2 ** 31 - 1,
                               init=m.bias.detach().clone()))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            params.append(dict(name=f"{name}.weight", mod=m, attr="weight", kind="float",
                               scale=None, init=m.weight.detach().clone()))
            params.append(dict(name=f"{name}.bias", mod=m, attr="bias", kind="float",
                               scale=None, init=m.bias.detach().clone()))
    for p in params:
        if p["kind"] == "quant":
            p["dz_int"] = torch.round(EPS / p["scale"])
    return params


def q_int(v, p):
    return torch.clamp(torch.round(v / p["scale"]), p["lo"], p["hi"])


def install(params, state, mode, delta_int=None, delta_float=None):
    for p in params:
        n = p["name"]
        if p["kind"] == "quant":
            iv = q_int(state[n], p) if mode == "master" else state[n]
            if delta_int is not None and n in delta_int:
                iv = torch.clamp(iv + delta_int[n], p["lo"], p["hi"])
            getattr(p["mod"], p["attr"]).data = (iv * p["scale"]).reshape(p["init"].shape).float()
        else:
            v = state[n]
            if delta_float is not None and n in delta_float:
                v = v + delta_float[n]
            getattr(p["mod"], p["attr"]).data = v.clone()


def draw_z(params, u):
    rng = np.random.RandomState(SEED + u)
    return {p["name"]: torch.from_numpy(
        (rng.randint(0, 2, size=tuple(p["init"].shape)).astype(np.float32) * 2.0 - 1.0))
        for p in params}


def probe_g(model, params, trX, trY, n_steps):
    """g at theta0 for n_steps probe draws (no updates). Returns list of g."""
    Xt = torch.from_numpy(trX)
    Yt = torch.from_numpy(trY)
    state = {p["name"]: p["init"].clone() for p in params}
    gs = []
    for u in range(n_steps):
        z = draw_z(params, u)
        idx = [(u * N_ACCUM + a) % N_TRAIN for a in range(N_ACCUM)]
        xb, yb = Xt[idx], Yt[idx]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                d = p["dz_int"] * z[n]
                di_p[n], di_m[n] = d, -d
            else:
                d = EPS * z[n]
                df_p[n], df_m[n] = d, -d
        with torch.no_grad():
            install(params, state, "master", di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            install(params, state, "master", di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
        gs.append((Lp - Lm) / (2.0 * EPS * N_ACCUM))
    install(params, state, "master")
    return gs


def run_regime(model, params, trX, trY, evX, evY, mode, lr, n_steps, tag):
    t0 = time.time()
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)
    if mode == "master":
        state = {p["name"]: p["init"].clone() for p in params}
    else:
        state = {p["name"]: (q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
                 for p in params}
    init_int = {p["name"]: q_int(p["init"], p) for p in params if p["kind"] == "quant"}

    def cur_int(n, p):
        return q_int(state[n], p) if mode == "master" else state[n]

    Xt, Yt = torch.from_numpy(trX), torch.from_numpy(trY)
    hist = dict(acc_step=[], acc_bal=[], train_loss=[])
    zero_move_steps = 0
    prev = {p["name"]: cur_int(p["name"], p).clone() for p in conv_w}

    for u in range(n_steps):
        z = draw_z(params, u)
        idx = [(u * N_ACCUM + a) % N_TRAIN for a in range(N_ACCUM)]
        xb, yb = Xt[idx], Yt[idx]
        di_p, di_m, df_p, df_m = {}, {}, {}, {}
        for p in params:
            n = p["name"]
            if p["kind"] == "quant":
                d = p["dz_int"] * z[n]
                di_p[n], di_m[n] = d, -d
            else:
                d = EPS * z[n]
                df_p[n], df_m[n] = d, -d
        with torch.no_grad():
            install(params, state, mode, di_p, df_p)
            Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            install(params, state, mode, di_m, df_m)
            Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
        coeff = -lr * (Lp - Lm) / (2.0 * EPS * N_ACCUM)
        for p in params:
            n = p["name"]
            if p["kind"] == "float" or mode == "master":
                state[n] = state[n] + coeff * z[n]
            else:
                delta = torch.round(coeff * z[n] / p["scale"])
                state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
        moved = 0
        for p in conv_w:
            ci = cur_int(p["name"], p)
            moved += int((ci != prev[p["name"]]).sum())
            prev[p["name"]] = ci.clone()
        if moved == 0:
            zero_move_steps += 1
        if (u + 1) % REG_EVAL_EVERY == 0 or u == n_steps - 1:
            with torch.no_grad():
                install(params, state, mode)
                bal, ov, _ = balanced(logits_of(model, evX), evY)
                tl = float(np.mean([float(F.cross_entropy(
                    model(Xt[i:i + 54]), Yt[i:i + 54], reduction="mean"))
                    for i in range(0, N_TRAIN, 54)]))
            hist["acc_step"].append(u + 1)
            hist["acc_bal"].append(bal)
            hist["train_loss"].append(tl)
            log(f"  [{tag}] step {u+1:4d}/{n_steps} loss={tl:.4f} bal={bal:.2f}% "
                f"({time.time()-t0:.0f}s)")

    with torch.no_grad():
        install(params, state, mode)
        bal, ov, rec = balanced(logits_of(model, evX), evY)
    cum = sum(int((cur_int(p["name"], p) != init_int[p["name"]]).sum()) for p in conv_w)
    return dict(mode=mode, lr=lr, n_steps=n_steps,
                balanced_accuracy=bal, overall_accuracy=ov,
                pct_steps_zero_conv_movement=100.0 * zero_move_steps / n_steps,
                cum_pct_convw_int8_changed=100.0 * cum / n_convw,
                history=hist)


def best_pooled_key(r):
    p = {k: v["balanced_accuracy"] for k, v in r["sweep"].items() if k.startswith("pooled@")}
    return max(p, key=p.get)


def stage_probe():
    r = load_results()
    probes = r.setdefault("probe", {})
    keys = [best_pooled_key(r), "old54"]
    preX, trX, trY, evX, evY = get_data()
    for key in keys:
        if key in probes:
            log(f"[probe] {key} done, skipping")
            continue
        t0 = time.time()
        _, model = build_model()
        freeze_config(model, r, key, trX)
        params = build_params(model)
        gs = np.asarray(probe_g(model, params, trX, trY, N_PROBE))
        s_w_ch = np.concatenate([p["scale"].reshape(-1).numpy()
                                 for p in params if p["name"].endswith(".conv.weight")])
        ent = {}
        for lr in (3e-6, 1e-5):
            upd = np.abs(lr * gs)[:, None] / s_w_ch[None, :]     # (steps, channels) in LSB
            ent[f"lr{lr:g}"] = dict(
                frac_step_channel_ge_half_LSB=float((upd >= 0.5).mean()),
                frac_steps_any_channel_ge_half=float((upd >= 0.5).any(1).mean()),
                mean_upd_LSB=float(upd.mean()), max_upd_LSB=float(upd.max()),
                median_upd_LSB=float(np.median(upd)))
        probes[key] = dict(
            n_probe=N_PROBE, eps=EPS,
            g_abs_mean=float(np.abs(gs).mean()), g_abs_median=float(np.median(np.abs(gs))),
            g_abs_max=float(np.abs(gs).max()),
            dL_abs_mean=float((np.abs(gs) * 2 * EPS * N_ACCUM).mean()),
            g_samples=gs.tolist(),
            s_w_channels=s_w_ch.tolist(),
            update_LSB=ent)
        save_results(r)
        log(f"[probe] {key}: |g| mean={probes[key]['g_abs_mean']:.3f} "
            f"clear@3e-6={ent['lr3e-06']['frac_step_channel_ge_half_LSB']:.4f} "
            f"({time.time()-t0:.0f}s)")


def stage_regimes():
    r = load_results()
    regs = r.setdefault("regimes", {})
    keys = [best_pooled_key(r), "old54"]
    preX, trX, trY, evX, evY = get_data()
    for key in keys:
        for mode in ("direct", "master"):
            tag = f"{key}|{mode}"
            if tag in regs:
                log(f"[regimes] {tag} done, skipping")
                continue
            _, model = build_model()
            freeze_config(model, r, key, trX)
            params = build_params(model)
            log(f"[regimes] {tag} lr={LR_FT:g} {N_REG} steps")
            regs[tag] = run_regime(model, params, trX, trY, evX, evY, mode, LR_FT, N_REG, tag)
            save_results(r)


# ---------------------------------------------------------------------------------------
def stage_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = load_results()
    sweep = r["sweep"]
    sites = list(next(iter(sweep.values()))["scales"].keys())
    configs = list(sweep.keys())

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for k in configs:
        ax.plot(range(len(sites)), [sweep[k]["scales"][s] for s in sites], "o-", ms=4, label=k)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([s.replace(".input_quant", ".in").replace(".output_quant", ".out")
                        for s in sites], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("frozen activation scale")
    ax.set_title("Activation scales per site per calibration config")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(HERE / "scales_per_site.png", dpi=140)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    xs = range(len(configs))
    for i, (m, ttl) in enumerate([("balanced_accuracy", "zero-shot bal-acc (batch 2) %"),
                                  ("logit_cosine_vs_float", "logit cosine vs float"),
                                  ("mean_abs_dCE_vs_float", "mean |dCE| vs float")]):
        v = [sweep[k][m] for k in configs]
        ax[i].bar(xs, v, color="#2f6fdb")
        for x, y in zip(xs, v):
            ax[i].text(x, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax[i].set_title(ttl)
        ax[i].grid(axis="y", alpha=.3)
    if "float_ref" in r:
        ax[0].axhline(r["float_ref"]["balanced_accuracy"], color="#1b7f4b", ls="--", lw=1,
                      label=f"float {r['float_ref']['balanced_accuracy']:.2f}%")
        ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "quality_metrics.png", dpi=140)

    if "probe" in r:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, pr in r["probe"].items():
            gs = np.abs(np.asarray(pr["g_samples"]))
            s_w = np.asarray(pr["s_w_channels"])
            upd = (3e-6 * gs)[:, None] / s_w[None, :]
            v = np.sort(upd.reshape(-1))
            ax.plot(v, np.linspace(0, 1, v.size), label=f"{key} (lr 3e-6)")
        ax.axvline(0.5, color="k", ls=":", lw=1, label="0.5 LSB threshold")
        ax.set_xscale("log")
        ax.set_xlabel("implied |update| per channel [LSB]")
        ax.set_ylabel("CDF over (step, channel)")
        ax.set_title("Direct-int8 update magnitude vs the rounding threshold")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(HERE / "lsb_clearance.png", dpi=140)

    if "regimes" in r:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for tag, run in r["regimes"].items():
            h = run["history"]
            ax.plot(h["acc_step"], h["acc_bal"], "o-", ms=3,
                    ls="-" if "master" in tag else "--", label=tag)
        for k in ("old54", best_pooled_key(r)):
            ax.axhline(r["sweep"][k]["balanced_accuracy"], lw=.8, alpha=.5, color="#888")
        ax.set_xlabel("ZO update step")
        ax.set_ylabel("batch-2 balanced accuracy (%)")
        ax.set_title(f"ZO fine-tune under best-pooled vs old calibration (lr {LR_FT:g})")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(HERE / "regime_trajectories.png", dpi=140)
    log("[plots] written")


# ---------------------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    stage = os.environ.get("STAGE", "all")
    todo = ["data", "pools", "sweep", "probe", "regimes", "plots"] if stage == "all" else [stage]
    log(f"\n==== run_study {time.strftime('%F %T')} stages={todo} ====")
    for s in todo:
        globals()[f"stage_{s}"]()
    log("==== run_study complete ====")


if __name__ == "__main__":
    main()
