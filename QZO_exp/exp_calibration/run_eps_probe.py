# SPDX-License-Identifier: MIT
"""Does a larger eps help the LSB stall? Theory: no — g = (L+-L-)/(2*eps), so eps cancels
to first order and the update lr*|g| is eps-invariant; larger eps only adds curvature bias
(and clamping) to the estimate, while too-small eps kills the PROBE (round(eps/s_w) -> 0).
This measures |g| and the implied update-LSB clearance at eps in {0.002, 0.005, 0.01(base),
0.05, 0.1}, pooled@99.99 calibration, same 100 shared-z probes at theta0."""
import numpy as np

import run_study as S

EPSES = [0.002, 0.005, 0.01, 0.05, 0.1]

def main():
    r = S.load_results()
    out = r.setdefault("eps_probe", {})
    _, trX, trY, evX, evY = S.get_data()
    base_eps = S.EPS
    for eps in EPSES:
        key = f"eps{eps:g}"
        if key in out:
            S.log(f"[eps] {key} done, skipping")
            continue
        S.EPS = eps
        try:
            _, model = S.build_model()
            S.freeze_config(model, r, "pooled@99.99", trX)
            params = S.build_params(model)          # dz_int = round(eps/s_w) recomputed
            dzs = np.concatenate([p["dz_int"].reshape(-1).numpy()
                                  for p in params if p["name"].endswith(".conv.weight")])
            gs = np.asarray(S.probe_g(model, params, trX, trY, 100))
            s_w = np.concatenate([p["scale"].reshape(-1).numpy()
                                  for p in params if p["name"].endswith(".conv.weight")])
            ent = {}
            for lr in (3e-6, 1e-5):
                upd = np.abs(lr * gs)[:, None] / s_w[None, :]
                ent[f"lr{lr:g}"] = dict(
                    mean_upd_LSB=float(upd.mean()),
                    frac_ge_half=float((upd >= 0.5).mean()))
            out[key] = dict(eps=eps, g_abs_mean=float(np.abs(gs).mean()),
                            g_abs_median=float(np.median(np.abs(gs))),
                            g_abs_max=float(np.abs(gs).max()),
                            probe_dz_int_min=int(dzs.min()), probe_dz_int_max=int(dzs.max()),
                            n_channels_dz_zero=int((dzs == 0).sum()),
                            update_LSB=ent)
            S.save_results(r)
            d = out[key]
            S.log(f"[eps] eps={eps:<6g} dz_int=[{d['probe_dz_int_min']},{d['probe_dz_int_max']}] "
                  f"(zero-dz ch={d['n_channels_dz_zero']})  |g| mean={d['g_abs_mean']:7.3f} "
                  f"max={d['g_abs_max']:7.3f}  updLSB@3e-6={ent['lr3e-06']['mean_upd_LSB']:.4f} "
                  f"clear={ent['lr3e-06']['frac_ge_half']:.4f}")
        finally:
            S.EPS = base_eps
    S.log("==== eps probe complete ====")

if __name__ == "__main__":
    main()
