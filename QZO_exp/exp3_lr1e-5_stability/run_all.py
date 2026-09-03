# SPDX-License-Identifier: MIT
"""Cross-fold stability: run QZO (direct int8, lr 1e-5) and float ZO (lr 3e-6) incremental
fine-tuning on folds 1 and 2, seed 42. Fold 3 is pulled from ../exp_calibration for the summary.
Resumable: completed folds are skipped. results.json here is local to exp3."""
import json
import time
from pathlib import Path

import numpy as np

import stability_lib as L
import run_study as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
FOLDS = [1, 2]
SEED = 42
# redirect run_study's logger to this experiment's run.log
S._LOG = open(HERE / "run.log", "a")


def load():
    return json.load(open(RESULTS)) if RESULTS.exists() else {}


def save(r):
    tmp = RESULTS.with_suffix(".json.tmp"); json.dump(r, open(tmp, "w"), indent=2); tmp.replace(RESULTS)


def run_one_fold(fold: int, r=None):
    r = r if r is not None else load()
    fk = f"fold{fold}"
    if fk in r.get("stability", {}):
        S.log(f"[fold {fold}] done, skipping"); return r
    S.SEED = SEED
    t0 = time.time()
    S.log(f"==== fold {fold}: calibrate + QZO + float ZO (seed {SEED}) ====")
    data = L.load_fold_data(fold)
    # QZO
    _, qm = L.build_qmodel(fold)
    L.calibrate_fold(fold, qm)
    qparams = S.build_params(qm)
    qzo = L.qzo_stream(qm, qparams, data, L.LR_QZO, tag=f"f{fold}")
    # float ZO
    fm = L.build_fmodel(fold)
    fp = L.fp_stream(fm, data, L.LR_FP, tag=f"f{fold}")
    r.setdefault("stability", {})[fk] = dict(qzo=qzo, fp=fp, seed=SEED)
    save(r)
    S.log(f"[fold {fold}] done ({time.time()-t0:.0f}s)")
    return r


def fold3_from_exp_calibration():
    """Pull fold-3 QZO/float from the exp_calibration results (same protocol/seed 42)."""
    p = L.EXP_CALIB / "results.json"
    if not p.exists():
        return None
    inc = json.load(open(p)).get("incremental")
    return inc if inc and "qzo" in inc and "fp" in inc else None


def summary(r):
    S.log("==== cross-fold summary (seed 42, mean b2-5) ====")
    f3 = fold3_from_exp_calibration()
    folds = {1: r["stability"]["fold1"], 2: r["stability"]["fold2"]}
    if f3:
        folds[3] = f3
    S.log(f"  {'fold':5s} {'QZO b2-5':>9s} {'float b2-5':>11s} {'Q-F':>6s}   per-batch QZO / float")
    for fold in sorted(folds):
        q = folds[fold]["qzo"]; f = folds[fold]["fp"]
        qb = [q[f"round{i}"]["acc_after"] for i in range(1, 5)]
        fb = [f[f"round{i}"]["acc_after"] for i in range(1, 5)]
        qm, fm = np.mean(qb), np.mean(fb)
        S.log(f"  {fold:<5d} {qm:9.2f} {fm:11.2f} {qm-fm:+6.2f}   "
              f"QZO[{','.join(f'{x:.1f}' for x in qb)}] float[{','.join(f'{x:.1f}' for x in fb)}]")
    S.log("  -- QZO conv weights moved per round (fraction of 14,880) --")
    for fold in sorted(folds):
        q = folds[fold]["qzo"]
        if "convw_moved_this_round_pct" not in q["round1"]:
            continue
        mv = [q[f"round{i}"]["convw_moved_this_round_pct"] for i in range(1, 5)]
        cum = q["round4"]["convw_net_cum_vs_pretrained_pct"]
        uni = q["round4"]["convw_union_ever_pct"]
        S.log(f"  fold {fold}: moved/round [{','.join(f'{x:.1f}%' for x in mv)}]  cum-net={cum:.1f}%  union={uni:.1f}%")
    r["stability_summary_written"] = True
    save(r)


def main():
    r = load()
    for fold in FOLDS:
        r = run_one_fold(fold, r)
    summary(r)
    S.log("==== stability complete ====")


if __name__ == "__main__":
    main()
