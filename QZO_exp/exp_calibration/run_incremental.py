# SPDX-License-Identifier: MIT
"""Incremental fine-tuning simulation, S01 / vocalized / fold 3, session 3.

Protocol (the on-device recipe): rounds r = 1..4. Each round fine-tunes on 54 stratified
windows (30% of the 180, seed 42) drawn from session-3 batch r, carries the weights, and
evaluates on the WHOLE next batch (r+1, 180 windows). 200 epochs x 54 / n_accum 4 = 2700
update steps per round, eps 0.01. The Rademacher schedule restarts at seed 42 each round
(each device round is a fresh runner invocation with --seed 42).

Two pipelines on the SAME per-round fixtures:
  qzo : direct int8, lr 1e-5, act scales frozen at pooled@99.99 (pretraining-data
        calibration), weight scales frozen at the pretrained per-channel abs-max —
        both held fixed across ALL rounds (no recalibration).
  fp  : plain float SpeechNet, float-ZO recipe lr 3e-6, same windows, eval-mode BN
        (frozen running stats), all conv w/b + BN gamma/beta + fc w/b trainable.

Per round we record acc before/after on batch r+1 and, for qzo, the conv int8 movement:
net within the round, cumulative net vs the pretrained weights, and the ever-moved union.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_study as S

ROUNDS = [1, 2, 3, 4]
STEPS = 200 * S.N_TRAIN // S.N_ACCUM          # 2700
LR_QZO = 1e-5
LR_FP = 3e-6
EVAL_EVERY = 675                              # 4 mid-round points
CACHE = S.HERE / "data_cache_incr.npz"


def stage_incr_data():
    if CACHE.exists():
        S.log("[incr-data] cache exists, skipping")
        return
    out = {}
    for r_ in ROUNDS:
        e = S.make_exporter(3, r_)
        ishape = e.get_input_shape()
        X, Y = e.get_data_source().load_batches(S.N_TRAIN, (1,) + tuple(ishape[1:]), 9,
                                                seed=S.SEED)
        out[f"trX{r_}"] = np.concatenate([np.asarray(a, np.float32) for a in X], 0)
        out[f"trY{r_}"] = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)
        ev = S.make_exporter(3, r_ + 1)
        i2, l2 = ev.get_data_source()._load_windows()
        out[f"evX{r_}"] = np.concatenate(i2, 0).astype(np.float32)
        out[f"evY{r_}"] = np.concatenate(l2, 0).astype(np.int64).reshape(-1)
        assert out[f"trX{r_}"].shape[0] == S.N_TRAIN and out[f"evX{r_}"].shape[0] == 180
        S.log(f"[incr-data] round {r_}: train batch{r_} {out[f'trX{r_}'].shape}  "
              f"eval batch{r_+1} {out[f'evX{r_}'].shape}")
    np.savez_compressed(CACHE, **out)


def bal_of(model, X, Y):
    with torch.no_grad():
        b, _, _ = S.balanced(S.logits_of(model, X), Y)
    return b


# ---------------------------------------------------------------------------------------
def qzo_incremental(res):
    d = np.load(CACHE)
    r = S.load_results()
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", None)   # act thresholds from results.json
    params = S.build_params(model)                    # weight scales frozen (pretrained)
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant"
                         else p["init"].clone()) for p in params}
    init_int = {p["name"]: state[p["name"]].clone() for p in conv_w}
    ever = {p["name"]: torch.zeros_like(state[p["name"]], dtype=torch.bool) for p in conv_w}

    # NOTE: rounds carry state, so resume granularity is the whole pipeline (see main) —
    # individual completed rounds cannot be skipped without replaying the state anyway.
    for r_ in ROUNDS:
        key = f"round{r_}"
        trX, trY = torch.from_numpy(d[f"trX{r_}"]), torch.from_numpy(d[f"trY{r_}"])
        evX, evY = d[f"evX{r_}"], d[f"evY{r_}"]
        with torch.no_grad():
            S.install(params, state, "direct")
        acc_before = bal_of(model, evX, evY)
        round_start = {p["name"]: state[p["name"]].clone() for p in conv_w}
        t0 = time.time()
        traj = []
        for u in range(STEPS):
            z = S.draw_z(params, u)                   # seed restarts each round
            idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
            xb, yb = trX[idx], trY[idx]
            di_p, di_m, df_p, df_m = {}, {}, {}, {}
            for p in params:
                n = p["name"]
                if p["kind"] == "quant":
                    dz = p["dz_int"] * z[n]
                    di_p[n], di_m[n] = dz, -dz
                else:
                    dz = S.EPS * z[n]
                    df_p[n], df_m[n] = dz, -dz
            with torch.no_grad():
                S.install(params, state, "direct", di_p, df_p)
                Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
                S.install(params, state, "direct", di_m, df_m)
                Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            coeff = -LR_QZO * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
            for p in params:
                n = p["name"]
                if p["kind"] == "float":
                    state[n] = state[n] + coeff * z[n]
                else:
                    delta = torch.round(coeff * z[n] / p["scale"])
                    if int(delta.abs().max()) and p["name"].endswith(".conv.weight"):
                        ever[n] |= (delta != 0)
                    state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
            if (u + 1) % EVAL_EVERY == 0:
                with torch.no_grad():
                    S.install(params, state, "direct")
                traj.append(dict(step=u + 1, bal=bal_of(model, evX, evY)))
        with torch.no_grad():
            S.install(params, state, "direct")
        acc_after = bal_of(model, evX, evY)
        net_round = sum(int((state[p["name"]] != round_start[p["name"]]).sum()) for p in conv_w)
        net_cum = sum(int((state[p["name"]] != init_int[p["name"]]).sum()) for p in conv_w)
        union = sum(int(e.sum()) for e in ever.values())
        res[key] = dict(
            train_batch=r_, eval_batch=r_ + 1, lr=LR_QZO, steps=STEPS,
            acc_before=acc_before, acc_after=acc_after, traj=traj,
            convw_moved_this_round_pct=100.0 * net_round / n_convw,
            convw_net_cum_vs_pretrained_pct=100.0 * net_cum / n_convw,
            convw_union_ever_pct=100.0 * union / n_convw)
        S.log(f"[qzo] round {r_}: b{r_+1} {acc_before:.2f}% -> {acc_after:.2f}%  "
              f"moved(this)={res[key]['convw_moved_this_round_pct']:.1f}%  "
              f"cum(net)={res[key]['convw_net_cum_vs_pretrained_pct']:.1f}%  "
              f"union={res[key]['convw_union_ever_pct']:.1f}%  ({time.time()-t0:.0f}s)")
    return res


# ---------------------------------------------------------------------------------------
def float_params_of(model):
    params = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            params.append(dict(name=f"{name}.weight", mod=m, attr="weight",
                               init=m.weight.detach().clone()))
            if m.bias is not None:
                params.append(dict(name=f"{name}.bias", mod=m, attr="bias",
                                   init=m.bias.detach().clone()))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            params.append(dict(name=f"{name}.weight", mod=m, attr="weight",
                               init=m.weight.detach().clone()))
            params.append(dict(name=f"{name}.bias", mod=m, attr="bias",
                               init=m.bias.detach().clone()))
    return params


def fp_logits(model, X):
    """Float SpeechNet forward is hardcoded to batch 1 — run sample by sample."""
    outs = []
    for i in range(X.shape[0]):
        outs.append(model(torch.from_numpy(X[i:i + 1])).detach())
    return torch.cat(outs, 0)


def fp_bal(model, X, Y):
    with torch.no_grad():
        logits = fp_logits(model, X).numpy()
    b, _, _ = S.balanced(logits, Y)
    return b


def fp_loss_sum(model, xb, yb):
    """Summed CE over a minibatch, forwarding one sample at a time (batch-1 model)."""
    logits = torch.cat([model(xb[i:i + 1]) for i in range(xb.shape[0])], 0)
    return F.cross_entropy(logits, yb, reduction="sum")


def fp_install(params, state, delta=None):
    for p in params:
        v = state[p["name"]]
        if delta is not None:
            v = v + delta[p["name"]]
        getattr(p["mod"], p["attr"]).data = v.clone()


def fp_draw_z(params, u):
    rng = np.random.RandomState(S.SEED + u)
    return {p["name"]: torch.from_numpy(
        (rng.randint(0, 2, size=tuple(p["init"].shape)).astype(np.float32) * 2.0 - 1.0))
        for p in params}


def fp_incremental(res):
    d = np.load(CACHE)
    model = S.build_float_model()
    params = float_params_of(model)
    S.log(f"[fp] trainable tensors: {len(params)}")
    state = {p["name"]: p["init"].clone() for p in params}
    for r_ in ROUNDS:
        key = f"round{r_}"
        trX, trY = torch.from_numpy(d[f"trX{r_}"]), torch.from_numpy(d[f"trY{r_}"])
        evX, evY = d[f"evX{r_}"], d[f"evY{r_}"]
        with torch.no_grad():
            fp_install(params, state)
        acc_before = fp_bal(model, evX, evY)
        t0 = time.time()
        for u in range(STEPS):
            z = fp_draw_z(params, u)
            idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
            xb, yb = trX[idx], trY[idx]
            dp = {n: S.EPS * z[n] for n in z}
            dm = {n: -v for n, v in dp.items()}
            with torch.no_grad():
                fp_install(params, state, dp)
                Lp = float(fp_loss_sum(model, xb, yb))
                fp_install(params, state, dm)
                Lm = float(fp_loss_sum(model, xb, yb))
            coeff = -LR_FP * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
            for p in params:
                state[p["name"]] = state[p["name"]] + coeff * z[p["name"]]
        with torch.no_grad():
            fp_install(params, state)
        acc_after = fp_bal(model, evX, evY)
        res[key] = dict(train_batch=r_, eval_batch=r_ + 1, lr=LR_FP, steps=STEPS,
                        acc_before=acc_before, acc_after=acc_after)
        S.log(f"[fp ] round {r_}: b{r_+1} {acc_before:.2f}% -> {acc_after:.2f}%  "
              f"({time.time()-t0:.0f}s)")
    return res


def main():
    stage_incr_data()
    r = S.load_results()
    incr = r.setdefault("incremental", {})
    if "qzo" not in incr:
        incr["qzo"] = qzo_incremental({})
        S.save_results(r)
    else:
        S.log("[qzo] all rounds present, skipping")
    if "fp" not in incr:
        incr["fp"] = fp_incremental({})
        S.save_results(r)
    else:
        S.log("[fp] all rounds present, skipping")
    S.log("==== incremental complete ====")
    for r_ in ROUNDS:
        q = incr["qzo"][f"round{r_}"]
        f = incr["fp"][f"round{r_}"]
        S.log(f"  round {r_} (ft b{r_} -> eval b{r_+1}): "
              f"QZO {q['acc_before']:.2f}->{q['acc_after']:.2f}%  "
              f"FP {f['acc_before']:.2f}->{f['acc_after']:.2f}%  "
              f"moved(this)={q['convw_moved_this_round_pct']:.1f}% "
              f"cum={q['convw_net_cum_vs_pretrained_pct']:.1f}%")

if __name__ == "__main__":
    main()
