# SPDX-License-Identifier: MIT
"""Why is quantized zero-shot (85.56%) > float zero-shot (81.67%) on session-3 batch 2?
Discriminating test: quant-vs-float on IN-DISTRIBUTION (pretraining sess 1+2) data vs the
HELD-OUT session 3. If quant wins only on held-out -> generalization/regularization
(consistent with pooled@99.99 clipping the activation outlier tail). If quant wins
everywhere -> not a generalization story. Also reports per-eval so we see consistency."""
import numpy as np
import torch

import run_study as S
import run_incremental as RI


def qbal(model, X, Y):
    with torch.no_grad():
        b, _, _ = S.balanced(S.logits_of(model, X), Y)
    return b


def load_eval(session, batch):
    e = S.make_exporter(session, batch)
    i, l = e.get_data_source()._load_windows()
    X = np.concatenate(i, 0).astype(np.float32)
    Y = np.concatenate(l, 0).astype(np.int64).reshape(-1)
    return X, Y


def main():
    r = S.load_results()
    # quantized model (pretrained weights, pooled@99.99 frozen scales)
    _, qm = S.build_model()
    S.freeze_config(qm, r, "pooled@99.99", None)
    # float model (same pretrained weights)
    fm = S.build_float_model()

    evals = [("in-dist  sess1 b1", 1, 1), ("in-dist  sess1 b3", 1, 3),
             ("in-dist  sess2 b2", 2, 2), ("held-out sess3 b2", 3, 2),
             ("held-out sess3 b3", 3, 3), ("held-out sess3 b4", 3, 4)]
    out = {}
    S.log("=== quant vs float zero-shot (pretrained weights, no FT) ===")
    S.log(f"  {'eval set':22s} {'quant':>7s} {'float':>7s} {'q-f':>7s}")
    for tag, sess, b in evals:
        X, Y = load_eval(sess, b)
        q = qbal(qm, X, Y)
        f = RI.fp_bal(fm, X, Y)
        out[tag] = dict(quant=q, float=f, diff=q - f, n=int(X.shape[0]))
        S.log(f"  {tag:22s} {q:7.2f} {f:7.2f} {q-f:+7.2f}")
        S.save_results({**r, "quant_vs_float_zeroshot": out}) if False else None
    r["quant_vs_float_zeroshot"] = out
    S.save_results(r)
    ind = [v["diff"] for k, v in out.items() if k.startswith("in-dist")]
    hel = [v["diff"] for k, v in out.items() if k.startswith("held")]
    S.log(f"  mean q-f  in-distribution = {np.mean(ind):+.2f}   held-out = {np.mean(hel):+.2f}")
    S.log("==== quant_vs_float_zeroshot complete ====")


if __name__ == "__main__":
    main()
