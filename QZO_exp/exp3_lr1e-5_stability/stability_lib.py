# SPDX-License-Identifier: MIT
"""Fold-parametrized helpers for the lr-1e-5 cross-fold stability study. Reuses the generic ZO
machinery from ../exp_calibration (run_study, run_incremental, calib_pooled); only the
fold-specific pieces (checkpoint, per-fold model/data, per-fold pooled@99.99 calibration) live
here. Fold k holds out session k; pretraining = the other two sessions (verified by zero-shot).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP_CALIB = (Path(__file__).resolve().parent.parent / "exp_calibration")
sys.path.insert(0, str(EXP_CALIB))

import run_study as S           # noqa: E402
import run_incremental as RI    # noqa: E402
import calib_pooled as CP       # noqa: E402

CKPT_DIR = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/"
            "S01/vocalized/speechnet/w1400ms/model_1")
STEPS = 200 * S.N_TRAIN // S.N_ACCUM     # 2700
LR_QZO = 1e-5
LR_FP = 3e-6
ALL_SESSIONS = (1, 2, 3)


def fold_ckpt(fold: int) -> str:
    return f"{CKPT_DIR}/leave_one_session_out_fold_{fold}.pt"


def pretrain_sessions(fold: int):
    return tuple(s for s in ALL_SESSIONS if s != fold)


# --- fold-specific model builders (set S.CKPT so S.make_exporter picks it up) --------------
def _set_fold(fold: int):
    S.CKPT = fold_ckpt(fold)


def build_qmodel(fold: int):
    _set_fold(fold)
    e = S.make_exporter(fold, 1)
    m = e.create_brevitas_model()
    m.eval()
    return e, m


def build_fmodel(fold: int):
    _set_fold(fold)
    e = S.make_exporter(fold, 1)
    fm = e.create_model()
    fm.eval()
    return fm


# --- per-fold pooled@99.99 activation calibration (on the fold's pretraining sessions) -----
def calibrate_fold(fold: int, model, log=S.log):
    _set_fold(fold)
    pre = []
    for sess in pretrain_sessions(fold):
        for b in range(1, 6):
            e = S.make_exporter(sess, b)
            xs, _ = e.get_data_source()._load_windows()
            x = np.concatenate(xs, 0).astype(np.float32)
            assert x.shape[0] == 180, f"fold{fold} sess{sess} b{b}: {x.shape[0]}!=180"
            pre.append(x)
    preX = np.concatenate(pre, 0)
    assert preX.shape[0] == 1800, f"fold{fold} pretrain {preX.shape[0]}!=1800"
    log(f"[calib] fold {fold}: pretrain sessions {pretrain_sessions(fold)}  {preX.shape}")
    coll, _ = CP.collect_pooled(model, preX, batch_size=64, cap=40_000_000, seed=0, log=log)
    th = CP.thresholds_from_pool(coll, [99.99])[99.99]
    thresholds = {site: d["threshold"] for site, d in th.items()}
    frozen = CP.freeze_act_thresholds(model, thresholds, log=log)
    return frozen


# --- per-fold incremental data (held-out session k, streaming b_r -> eval b_{r+1}) ---------
def load_fold_data(fold: int):
    _set_fold(fold)
    out = {}
    for r_ in (1, 2, 3, 4):
        e = S.make_exporter(fold, r_)
        ishape = e.get_input_shape()
        X, Y = e.get_data_source().load_batches(S.N_TRAIN, (1,) + tuple(ishape[1:]), 9,
                                                seed=S.SEED)
        out[f"trX{r_}"] = np.concatenate([np.asarray(a, np.float32) for a in X], 0)
        out[f"trY{r_}"] = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)
        ev = S.make_exporter(fold, r_ + 1)
        i2, l2 = ev.get_data_source()._load_windows()
        out[f"evX{r_}"] = np.concatenate(i2, 0).astype(np.float32)
        out[f"evY{r_}"] = np.concatenate(l2, 0).astype(np.int64).reshape(-1)
        assert out[f"trX{r_}"].shape[0] == S.N_TRAIN and out[f"evX{r_}"].shape[0] == 180
    return out


# --- QZO streaming (direct int8, carry state, per-round movement) --------------------------
def qzo_stream(model, params, data, lr, log=S.log, tag=""):
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    init_int = {p["name"]: state[p["name"]].clone() for p in conv_w}
    ever = {p["name"]: torch.zeros_like(state[p["name"]], dtype=torch.bool) for p in conv_w}
    rounds = {}
    for r_ in (1, 2, 3, 4):
        trX = torch.from_numpy(data[f"trX{r_}"]); trY = torch.from_numpy(data[f"trY{r_}"])
        evX, evY = data[f"evX{r_}"], data[f"evY{r_}"]
        with torch.no_grad():
            S.install(params, state, "direct")
            ab, _, _ = S.balanced(S.logits_of(model, evX), evY)
        rstart = {p["name"]: state[p["name"]].clone() for p in conv_w}
        for u in range(STEPS):
            z = S.draw_z(params, u)
            idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
            xb, yb = trX[idx], trY[idx]
            di_p, di_m, df_p, df_m = {}, {}, {}, {}
            for p in params:
                n = p["name"]
                if p["kind"] == "quant":
                    dz = p["dz_int"] * z[n]; di_p[n], di_m[n] = dz, -dz
                else:
                    dz = S.EPS * z[n]; df_p[n], df_m[n] = dz, -dz
            with torch.no_grad():
                S.install(params, state, "direct", di_p, df_p)
                Lp = float(F.cross_entropy(model(xb), yb, reduction="sum"))
                S.install(params, state, "direct", di_m, df_m)
                Lm = float(F.cross_entropy(model(xb), yb, reduction="sum"))
            coeff = -lr * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
            for p in params:
                n = p["name"]
                if p["kind"] == "float":
                    state[n] = state[n] + coeff * z[n]
                else:
                    delta = torch.round(coeff * z[n] / p["scale"])
                    if n.endswith(".conv.weight"):
                        ever[n] |= (delta != 0)
                    state[n] = torch.clamp(state[n] + delta, p["lo"], p["hi"])
        with torch.no_grad():
            S.install(params, state, "direct")
            aa, _, _ = S.balanced(S.logits_of(model, evX), evY)
        net_r = sum(int((state[p["name"]] != rstart[p["name"]]).sum()) for p in conv_w)
        net_c = sum(int((state[p["name"]] != init_int[p["name"]]).sum()) for p in conv_w)
        uni = sum(int(e.sum()) for e in ever.values())
        rounds[f"round{r_}"] = dict(
            train_batch=r_, eval_batch=r_ + 1, acc_before=ab, acc_after=aa,
            convw_moved_this_round_pct=100.0 * net_r / n_convw,
            convw_net_cum_vs_pretrained_pct=100.0 * net_c / n_convw,
            convw_union_ever_pct=100.0 * uni / n_convw)
        log(f"[qzo] {tag} r{r_}: b{r_+1} {ab:.2f}->{aa:.2f}%  moved(this)={100.0*net_r/n_convw:.1f}%")
    return rounds


# --- float ZO streaming (reuse RI's float helpers) ----------------------------------------
def fp_stream(fmodel, data, lr, log=S.log, tag=""):
    params = RI.float_params_of(fmodel)
    state = {p["name"]: p["init"].clone() for p in params}
    rounds = {}
    for r_ in (1, 2, 3, 4):
        trX = torch.from_numpy(data[f"trX{r_}"]); trY = torch.from_numpy(data[f"trY{r_}"])
        evX, evY = data[f"evX{r_}"], data[f"evY{r_}"]
        with torch.no_grad():
            RI.fp_install(params, state)
        ab = RI.fp_bal(fmodel, evX, evY)
        for u in range(STEPS):
            z = RI.fp_draw_z(params, u)
            idx = [(u * S.N_ACCUM + a) % S.N_TRAIN for a in range(S.N_ACCUM)]
            xb, yb = trX[idx], trY[idx]
            dp = {n: S.EPS * z[n] for n in z}; dm = {n: -v for n, v in dp.items()}
            with torch.no_grad():
                RI.fp_install(params, state, dp)
                Lp = float(RI.fp_loss_sum(fmodel, xb, yb))
                RI.fp_install(params, state, dm)
                Lm = float(RI.fp_loss_sum(fmodel, xb, yb))
            coeff = -lr * (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)
            for p in params:
                state[p["name"]] = state[p["name"]] + coeff * z[p["name"]]
        with torch.no_grad():
            RI.fp_install(params, state)
        aa = RI.fp_bal(fmodel, evX, evY)
        rounds[f"round{r_}"] = dict(train_batch=r_, eval_batch=r_ + 1,
                                    acc_before=ab, acc_after=aa)
        log(f"[fp ] {tag} r{r_}: b{r_+1} {ab:.2f}->{aa:.2f}%")
    return rounds
