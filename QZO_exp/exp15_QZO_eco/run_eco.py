# SPDX-License-Identifier: MIT
"""exp15 — ECO (error-feedback via momentum) for on-device quantized ZO fine-tuning.

Transfer of arXiv:2601.22101 (ECO) to our SpeechNet INT8 ZO setting (see
../../../docs/eco_paper_summary.md §8). Question: at the natural small learning rate
(3e-6) the direct-int8 ZO update STALLS — 100% of steps move zero conv weights, 0% of
conv weights ever change, accuracy frozen at zero-shot. The master-weight version moves
weights but costs a full FP32 shadow buffer. ECO injects the per-step rounding residual
into the SGD-momentum buffer, so momentum doubles as the error-feedback accumulator and
the sub-LSB updates are no longer discarded — no master weights, no extra buffer.

We reuse the faithful Brevitas SpeechNet + the incremental FT protocol from
exp_calibration (run_study `S`, run_incremental `RI`): same pretrained init, frozen
pooled@99.99 act scales + frozen pretrained per-channel weight scales, same 4-round
carry (ft batch r -> eval batch r+1), same z-seed schedule (restart seed 42 each round),
same 54 stratified windows, n_accum 4, eps 0.01.

Update rules (per quantized weight, in the dequantized/real domain):
  g       = (Lp - Lm)/(2*eps*n_accum) * z          # ZO gradient estimate (real)
  m_tilde = beta*m + (1-beta)*g                     # SGDM first moment
  th_hat  = w_int * s_w                             # current on-grid weight (real)
  th_tilde= th_hat - lr*m_tilde                     # tentative real step
  w_new   = clamp(quant(th_tilde/s_w), lo, hi)      # RTN or SR
  e       = th_tilde - w_new*s_w                    # rounding residual (real)
  ef=memoryfree : m = m_tilde + (1/lr)*(1 - 1/beta)*e     # ECO Alg.2 (no extra memory)
  ef=exact      : m = m_tilde + (1/lr)*e_prev - (1/(lr*beta))*e ; e_prev=e   # §2.2 (stores e)
  ef=none       : m = m_tilde                              # plain SGDM, no EF (ablation)
Float params (BN gamma/beta) use plain SGDM (no quant, no residual).

Metrics per round (paper §8.6): post-adaptation balanced acc; % steps with any conv
movement; % conv weights moved (this round / cum vs pretrained / ever-union); steps-to-
first-move distribution; and the ECO go/no-go diagnostic cos(e_t, e_{t+1}) and
||e_{t+1}||/||e_t||.

Env:
  ARM     one of the ARMS keys, or "all_diag" (β-sweep diagnostic) / "all_incr".
  STEPS   override steps/round (default 2700). ROUNDS  e.g. "1" or "1,2,3,4".
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
CALIB = HERE.parent / "exp_calibration"
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[1]))          # Onnx4Deeploy root

import run_study as S            # noqa: E402
import run_incremental as RI     # noqa: E402

RESULTS = HERE / "results.json"
_LOG = open(HERE / "run.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


def load_results():
    return json.load(open(RESULTS)) if RESULTS.exists() else {}


def save_results(r):
    tmp = RESULTS.with_suffix(".json.tmp")
    json.dump(r, open(tmp, "w"), indent=2)
    tmp.replace(RESULTS)


# --------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------
# ef: memoryfree | exact | none ;  round: rtn | sr ;  eco True enables momentum+injection.
ARMS = {
    # baselines (no momentum): the stall and the working reference
    "direct_3e6":      dict(eco=False, lr=3e-6),                       # reproduce the stall
    "direct_1e5":      dict(eco=False, lr=1e-5),                       # working direct ref (~86%)
    # ECO memory-free at the natural small lr where direct stalls
    "eco_mf_3e6_b90":      dict(eco=True, ef="memoryfree", rnd="rtn", lr=3e-6, beta=0.90),
    "eco_mf_3e6_b99":      dict(eco=True, ef="memoryfree", rnd="rtn", lr=3e-6, beta=0.99),
    "eco_mf_3e6_b90_sr":   dict(eco=True, ef="memoryfree", rnd="sr",  lr=3e-6, beta=0.90),
    # exact error feedback (stores e) — the fallback if e_t/e_{t+1} decorrelate
    "eco_ex_3e6_b90":      dict(eco=True, ef="exact",      rnd="rtn", lr=3e-6, beta=0.90),
    # ablation: momentum but NO error feedback (should still stall — rounding discards sub-LSB)
    "sgdm_none_3e6_b90":   dict(eco=True, ef="none",       rnd="rtn", lr=3e-6, beta=0.90),
}


def sr_round(x, rng):
    """Unbiased stochastic rounding: floor(x + U), U~Uniform[0,1)."""
    return torch.floor(x + torch.from_numpy(rng.random_sample(tuple(x.shape)).astype(np.float32)))


def eco_incremental(cfg, res, rounds, steps, eval_every=675):
    """One arm over the incremental rounds. cfg keys: eco, lr, [ef, rnd, beta]."""
    d = np.load(RI.CACHE)
    r = S.load_results()
    _, model = S.build_model()
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    conv_w = [p for p in params if p["kind"] == "quant" and p["name"].endswith(".conv.weight")]
    n_convw = sum(int(p["init"].numel()) for p in conv_w)

    eco = cfg["eco"]
    lr = cfg["lr"]
    beta = cfg.get("beta", 0.0)
    ef = cfg.get("ef", "none")
    rnd = cfg.get("rnd", "rtn")
    inj = (1.0 / lr) * (1.0 - 1.0 / beta) if (eco and ef != "none") else 0.0   # ECO alpha

    # state: quant -> int8 code; float -> real value
    state = {p["name"]: (S.q_int(p["init"], p) if p["kind"] == "quant" else p["init"].clone())
             for p in params}
    mom = {p["name"]: torch.zeros_like(p["init"]) for p in params}            # momentum (real)
    eprev = {p["name"]: torch.zeros_like(p["init"]) for p in params}          # exact-EF store
    init_int = {p["name"]: state[p["name"]].clone() for p in conv_w}
    ever = {p["name"]: torch.zeros_like(state[p["name"]], dtype=torch.bool) for p in conv_w}
    first_move = {p["name"]: torch.full_like(state[p["name"]], -1, dtype=torch.int32)
                  for p in conv_w}
    srng = np.random.RandomState(S.SEED + 777)     # SR noise stream (independent of z)

    for r_ in rounds:
        key = f"round{r_}"
        trX, trY = torch.from_numpy(d[f"trX{r_}"]), torch.from_numpy(d[f"trY{r_}"])
        evX, evY = d[f"evX{r_}"], d[f"evY{r_}"]
        with torch.no_grad():
            S.install(params, state, "direct")
        acc_before = RI.bal_of(model, evX, evY)
        round_start = {p["name"]: state[p["name"]].clone() for p in conv_w}
        # momentum restarts each round (each device round is a fresh runner invocation)
        for n in mom:
            mom[n].zero_()
            eprev[n].zero_()
        t0 = time.time()
        traj, zero_move_steps = [], 0
        cos_list, ratio_list, global_step = [], [], 0
        prev_e = None
        for u in range(steps):
            z = S.draw_z(params, u)
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
            gscalar = (Lp - Lm) / (2.0 * S.EPS * S.N_ACCUM)

            moved = 0
            e_vec = []                      # rounding residuals over conv weights this step
            for p in params:
                n = p["name"]
                g = gscalar * z[n]
                if p["kind"] == "float":
                    if eco:
                        mom[n] = beta * mom[n] + (1.0 - beta) * g
                        state[n] = state[n] - lr * mom[n]
                    else:
                        state[n] = state[n] - lr * g
                    continue
                # quantized weight/bias
                s_w = p["scale"]
                if eco:
                    mtil = beta * mom[n] + (1.0 - beta) * g
                    th_tilde = state[n] * s_w - lr * mtil
                    q = th_tilde / s_w
                    w_new = sr_round(q, srng) if rnd == "sr" else torch.round(q)
                    w_new = torch.clamp(w_new, p["lo"], p["hi"])
                    e = th_tilde - w_new * s_w
                    if ef == "memoryfree":
                        mom[n] = mtil + inj * e
                    elif ef == "exact":
                        mom[n] = mtil + (1.0 / lr) * eprev[n] - (1.0 / (lr * beta)) * e
                        eprev[n] = e
                    else:                    # none
                        mom[n] = mtil
                else:
                    delta = torch.round(gscalar * z[n] * (-lr) / s_w)  # == round(coeff*z/s_w)
                    w_new = torch.clamp(state[n] + delta, p["lo"], p["hi"])
                    e = None
                changed = (w_new != state[n])
                if p["name"].endswith(".conv.weight"):
                    moved += int(changed.sum())
                    ever[n] |= changed
                    newly = changed & (first_move[n] < 0)
                    first_move[n][newly] = global_step
                    if e is not None:
                        e_vec.append(e.reshape(-1))
                state[n] = w_new
            if moved == 0:
                zero_move_steps += 1
            # ECO go/no-go diagnostic: cos(e_t, e_{t+1}) and norm ratio over conv residuals
            if e_vec:
                ev = torch.cat(e_vec)
                if prev_e is not None:
                    npv, ncv = float(prev_e.norm()), float(ev.norm())
                    if npv > 0 and ncv > 0:
                        cos_list.append(float(torch.dot(prev_e, ev) / (npv * ncv)))
                        ratio_list.append(ncv / npv)
                prev_e = ev
            global_step += 1
            if (u + 1) % eval_every == 0:
                with torch.no_grad():
                    S.install(params, state, "direct")
                traj.append(dict(step=u + 1, bal=RI.bal_of(model, evX, evY)))
        with torch.no_grad():
            S.install(params, state, "direct")
        acc_after = RI.bal_of(model, evX, evY)
        net_round = sum(int((state[p["name"]] != round_start[p["name"]]).sum()) for p in conv_w)
        net_cum = sum(int((state[p["name"]] != init_int[p["name"]]).sum()) for p in conv_w)
        union = sum(int(e.sum()) for e in ever.values())
        fm_all = torch.cat([first_move[p["name"]].reshape(-1) for p in conv_w])
        fm_moved = fm_all[fm_all >= 0].float()
        res[key] = dict(
            train_batch=r_, eval_batch=r_ + 1, lr=lr, beta=beta, ef=ef, rnd=rnd, steps=steps,
            acc_before=acc_before, acc_after=acc_after, traj=traj,
            pct_steps_any_conv_move=100.0 * (steps - zero_move_steps) / steps,
            convw_moved_this_round_pct=100.0 * net_round / n_convw,
            convw_net_cum_vs_pretrained_pct=100.0 * net_cum / n_convw,
            convw_union_ever_pct=100.0 * union / n_convw,
            steps_to_first_move_median=float(fm_moved.median()) if fm_moved.numel() else None,
            steps_to_first_move_p90=float(fm_moved.quantile(0.9)) if fm_moved.numel() else None,
            cos_e_mean=float(np.mean(cos_list)) if cos_list else None,
            cos_e_median=float(np.median(cos_list)) if cos_list else None,
            norm_ratio_mean=float(np.mean(ratio_list)) if ratio_list else None)
        rr = res[key]
        log(f"  [{cfg.get('_tag','?')}] r{r_} b{r_+1} {acc_before:.2f}->{acc_after:.2f}%  "
            f"any-move-steps={rr['pct_steps_any_conv_move']:.1f}%  "
            f"moved(this)={rr['convw_moved_this_round_pct']:.1f}%  "
            f"union={rr['convw_union_ever_pct']:.1f}%  "
            f"cos(e)={rr['cos_e_mean']}  1stmove_med={rr['steps_to_first_move_median']}  "
            f"({time.time()-t0:.0f}s)")
    return res


def run_arm(name, rounds, steps):
    cfg = dict(ARMS[name])
    cfg["_tag"] = name
    r = load_results()
    arms = r.setdefault("arms", {})
    if name in arms and all(f"round{x}" in arms[name] for x in rounds):
        log(f"[{name}] already complete for rounds {rounds}, skipping")
        return
    log(f"==== arm {name}: {cfg}  rounds={rounds} steps={steps} ====")
    arms[name] = eco_incremental(cfg, arms.get(name, {}), rounds, steps)
    save_results(r)


def main():
    torch.manual_seed(S.SEED)
    np.random.seed(S.SEED)
    RI.stage_incr_data()
    steps = int(os.environ.get("STEPS", RI.STEPS))
    rounds = [int(x) for x in os.environ.get("ROUNDS", "1,2,3,4").split(",")]
    arm = os.environ.get("ARM", "all_incr")
    log(f"\n#### run_eco {time.strftime('%F %T')} ARM={arm} steps={steps} rounds={rounds} ####")
    if arm == "all_diag":
        for a in ["eco_mf_3e6_b90", "eco_mf_3e6_b99", "sgdm_none_3e6_b90"]:
            run_arm(a, rounds, steps)
    elif arm == "all_incr":
        for a in ["direct_3e6", "direct_1e5", "eco_mf_3e6_b90", "eco_mf_3e6_b99",
                  "eco_mf_3e6_b90_sr", "eco_ex_3e6_b90", "sgdm_none_3e6_b90"]:
            run_arm(a, rounds, steps)
    else:
        run_arm(arm, rounds, steps)
    log("#### run_eco done ####")


if __name__ == "__main__":
    main()
