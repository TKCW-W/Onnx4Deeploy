# SPDX-License-Identifier: MIT
"""Meeting-prep (fork session, 2026-09-04): DEVICE-FAITHFUL (fc-float) QZO incremental
fine-tuning, fold 3, rounds 1..4 (ft batch r -> eval batch r+1), direct int8 @ lr 1e-5,
canonical pooled@99.99 scales, seed as given.

Everything imported read-only from exp_calibration / exp3 / exp9-pytorch_ref; results and log
go ONLY into this directory (no writes to any shared results.json / run.log — the original
session owns those).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp3_lr1e-5_stability")
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/pytorch_ref")

import run_study as S            # noqa: E402
import stability_lib as L        # noqa: E402
import run_fc_float_ref as RF    # noqa: E402

S._LOG = open(HERE / "run.log", "a")          # redirect logging away from shared files
RESULTS = HERE / "results_fcfloat.json"
SEEDS = [int(x) for x in (sys.argv[1:] or ["42"])]


def run_seed(seed: int):
    S.SEED = seed
    r = S.load_results()                       # read-only (canonical pooled thresholds)
    data = L.load_fold_data(3)                 # seed-dependent 54-window draws, whole-batch evals
    _, model = L.build_qmodel(3)
    S.freeze_config(model, r, "pooled@99.99", None)
    params = S.build_params(model)
    params = RF.to_float_fc(model, params)     # DEVICE-FAITHFUL: float fc head, fc_iq bypassed
    S.log(f"==== fc-float incremental fold3, seed {seed}, lr {L.LR_QZO:g} ====")
    rounds = L.qzo_stream(model, params, data, L.LR_QZO, tag=f"fcfloat-s{seed}")
    return rounds


def main():
    out = json.load(open(RESULTS)) if RESULTS.exists() else {}
    for seed in SEEDS:
        k = f"seed{seed}"
        if k in out:
            S.log(f"[fcfloat] {k} done, skipping")
            continue
        out[k] = run_seed(seed)
        json.dump(out, open(RESULTS, "w"), indent=1)
    S.log("==== fcfloat incremental complete ====")
    for k, rr in out.items():
        accs = [rr[f"round{i}"]["acc_after"] for i in range(1, 5)]
        mv = [rr[f"round{i}"]["convw_moved_this_round_pct"] for i in range(1, 5)]
        S.log(f"  {k}: b2-5 after = {[round(a,2) for a in accs]}  mean={sum(accs)/4:.2f}  "
              f"moved/round = {[round(m,1) for m in mv]}")


if __name__ == "__main__":
    main()
