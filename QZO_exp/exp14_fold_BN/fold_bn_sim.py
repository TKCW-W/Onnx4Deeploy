# SPDX-License-Identifier: MIT
"""exp14: QZO with BatchNorm FOLDED into the quantized conv (supervisor's design) — PyTorch/Brevitas simulation.
Reuses the exp_masterweight_ft harness verbatim (mw.py copy: build = create model -> real-data calibration -> freeze
scales -> trainable params; run_regime; evaluate) and inserts ONE step: right after the pretrained Brevitas model is
created and BEFORE calibration, fold each block's fp32 BN (running stats, gamma, beta) into the block's QuantConv2d
weight/bias and replace the BN by Identity. Weight scales are then re-derived from the folded weights and activation
scales re-calibrated on the folded model. Trainable params = int8 conv W'/b' (+ fc), no fp32 BN params.
Run in agitated_hugle:  EXTRA=x python3 fold_bn_sim.py   (EXTRA non-empty keeps mw's log in append mode)"""
import json, os, sys, time
os.environ.setdefault("EXTRA", "x")
sys.path.insert(0, "/app/Onnx4Deeploy"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import mw                                                   # the harness (copied), HERE = this dir
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter

FOLD_LOG = []
def fold_bn_(model):
    n = 0
    for b in model.blocks:
        bn, conv = b.bn, b.conv
        if not isinstance(bn, nn.BatchNorm2d): continue
        with torch.no_grad():
            inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)          # gamma / sigma
            W = conv.weight; conv.weight.copy_(W * inv.reshape(-1, 1, 1, 1))
            bias = conv.bias if conv.bias is not None else torch.zeros(W.shape[0])
            new_b = (bias - bn.running_mean) * inv + bn.bias
            if conv.bias is None: conv.bias = nn.Parameter(new_b)
            else: conv.bias.copy_(new_b)
        FOLD_LOG.append(f"   folded block: gamma/sigma in [{float(inv.min()):.3f},{float(inv.max()):.3f}]")
        b.bn = nn.Identity(); n += 1
    return n

_orig = SpeechNetExporter.create_brevitas_model
def _create_folded(self, *a, **k):
    m = _orig(self, *a, **k); n = fold_bn_(m); mw.log(f"[exp14] folded {n} BN layers into conv weight/bias; BN -> Identity"); [mw.log(l) for l in FOLD_LOG]
    return m
SpeechNetExporter.create_brevitas_model = _create_folded

def build_folded(calib: str):
    """mw.build() with (a) the fold (patched constructor) and (b) an optional pooled-percentile activation
    calibration applied BEFORE the trainable grids are built (the int32 bias grid is s_in*s_w)."""
    import brevitas.nn as qnn
    from brevitas.graph.calibrate import calibration_mode
    e1 = mw.make_exporter(1); ishape = e1.get_input_shape()
    X, Y = e1.get_data_source().load_batches(mw.N_TRAIN, (1,) + tuple(ishape[1:]), e1.config["num_classes"], seed=mw.SEED)
    trX = np.concatenate([np.asarray(a, np.float32) for a in X], 0); trY = np.asarray([int(np.asarray(y).reshape(-1)[0]) for y in Y], np.int64)
    e2 = mw.make_exporter(2); i2, l2 = e2.get_data_source()._load_windows(); evX = np.concatenate(i2, 0); evY = np.concatenate(l2, 0)
    model = e1.create_brevitas_model(); model.eval()
    FC_FLOAT = os.environ.get("FC_FLOAT", "0") == "1"
    if FC_FLOAT:   # device design: fc in fp32 on the fp32 GAP output (no fc input quant), trained as a smooth float param
        lin = nn.Linear(model.fc.in_features, model.fc.out_features)
        lin.weight.data = model.fc.weight.detach().clone(); lin.bias.data = model.fc.bias.detach().clone()
        model.fc = lin; model.fc_iq = nn.Identity(); mw.log("[exp14] FC_FLOAT: fc -> fp32 nn.Linear, fc input quant removed")
    with torch.no_grad(), calibration_mode(model): model(torch.from_numpy(trX[:mw.CALIB_SAMPLES]))
    model.eval()
    if calib.startswith("pooled@"):
        sys.path.insert(0, "/app/Onnx4Deeploy/QZO_exp/exp_calibration")
        from calib_pooled import collect_pooled, freeze_act_thresholds
        pct = float(calib.split("@")[1]); coll, meta = collect_pooled(model, trX)
        th = {n: coll.quantile(n, pct) for n in coll.sites}
        fr = freeze_act_thresholds(model, th, log=mw.log); mw.log(f"[exp14] act thresholds frozen at pooled@{pct:g} on the {len(trX)} training windows ({len(fr)} sites)")
    params = []
    for name, m in [(n, m) for n, m in model.named_modules() if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear))]:
        s_w = m.quant_weight().scale.detach().clone(); s_in = m.input_quant.scale().detach().clone()
        m.weight_quant.tensor_quant.scaling_impl = mw.ConstScale(s_w)
        params.append(dict(name=f"{name}.weight", mod=m, attr="weight", kind="quant", scale=s_w, lo=-127, hi=127, init=m.weight.detach().clone()))
        s_b = (s_in * s_w).reshape(-1)
        params.append(dict(name=f"{name}.bias", mod=m, attr="bias", kind="quant", scale=s_b, lo=-(2 ** 31) + 1, hi=2 ** 31 - 1, init=m.bias.detach().clone()))
    for p in params: p["dz_int"] = torch.round(mw.EPS / p["scale"])
    if FC_FLOAT:
        params.append(dict(name="fc.weight", mod=model.fc, attr="weight", kind="float", scale=None, init=model.fc.weight.detach().clone()))
        params.append(dict(name="fc.bias", mod=model.fc, attr="bias", kind="float", scale=None, init=model.fc.bias.detach().clone()))
    mw.log(f"trainable params: {len(params)} (all quantized; BN folded)")
    return model, params, trX, trY, evX, evY, mw.act_scales(model)

def run_regime_lsb(model, params, trX, trY, evX, evY, mode, lr, K, n_steps=None, tag=""):
    """Scale-invariant LSB-domain ZO (exp14): every quantized weight is perturbed by K of ITS OWN quant steps
    (d = K*z LSB), g = (L+ - L-)/(2*K*n_accum), update = round(-lr*g*z) LSB (direct_lsb: int grid is the state;
    master_lsb: latent fp32 in LSB units, read out by round). Same z draw (mw.draw_z) as the other regimes."""
    import torch.nn.functional as TF
    n_steps = n_steps or mw.N_STEPS; t0 = time.time()
    Q = [p for p in params if p["kind"] == "quant"]; FL = [p for p in params if p["kind"] == "float"]; lr_f = float(os.environ.get("LR_F", "1e-5"))
    state = {p["name"]: mw.q_int(p["init"], p) for p in Q}; state.update({p["name"]: p["init"].clone() for p in FL}); init_int = {p["name"]: state[p["name"]].clone() for p in Q}
    lat = {p["name"]: state[p["name"]].clone() for p in Q} if mode == "master_lsb" else None
    Xt, Yt = torch.from_numpy(trX), torch.from_numpy(trY); zero_steps = 0
    conv_w = [p["name"] for p in params if p["name"].startswith("blocks") and p["name"].endswith(".conv.weight")]
    for u in range(n_steps):
        z = mw.draw_z(params, u); idx = [(u * mw.N_ACCUM + a) % mw.N_TRAIN for a in range(mw.N_ACCUM)]; xb, yb = Xt[idx], Yt[idx]
        cur = dict(state) if lat is None else {**{p["name"]: torch.clamp(torch.round(lat[p["name"]]), p["lo"], p["hi"]) for p in Q}, **{p["name"]: state[p["name"]] for p in FL}}
        di_p = {p["name"]: K * z[p["name"]] for p in Q}; df_p = {p["name"]: mw.EPS * z[p["name"]] for p in FL}
        with torch.no_grad():
            mw.install(params, cur, "direct", di_p, df_p); Lp = float(TF.cross_entropy(model(xb), yb, reduction="sum"))
            mw.install(params, cur, "direct", {n: -v for n, v in di_p.items()}, {n: -v for n, v in df_p.items()}); Lm = float(TF.cross_entropy(model(xb), yb, reduction="sum"))
        g = (Lp - Lm) / (2.0 * K * mw.N_ACCUM); g_abs = (Lp - Lm) / (2.0 * mw.EPS * mw.N_ACCUM); moved = 0
        for p in FL: state[p["name"]] = state[p["name"]] + (-lr_f * g_abs) * z[p["name"]]
        for p in Q:
            n = p["name"]
            if lat is None:
                d = torch.round(-lr * g * z[n]); state[n] = torch.clamp(state[n] + d, p["lo"], p["hi"]); moved += int((d != 0).sum())
            else: lat[n] = torch.clamp(lat[n] + (-lr * g * z[n]), p["lo"], p["hi"])
        if lat is None and moved == 0: zero_steps += 1
        if u % 300 == 0: mw.log(f"     [{tag}] u{u} L+={Lp:.4f} L-={Lm:.4f} g={g:+.4f}")
    final = dict(state) if lat is None else {**{p["name"]: torch.clamp(torch.round(lat[p["name"]]), p["lo"], p["hi"]) for p in Q}, **{p["name"]: state[p["name"]] for p in FL}}
    with torch.no_grad(): mw.install(params, final, "direct", None, None)
    acc = mw.evaluate(model, evX, evY); tot = sum(int(init_int[n].numel()) for n in conv_w)
    mv = 100.0 * sum(int((final[n] != init_int[n]).sum()) for n in conv_w) / tot
    r = dict(mode=mode, lr_lsb=lr, K_lsb=K, lr_f=lr_f if FL else None, n_steps=n_steps, balanced_accuracy=acc["balanced_accuracy"], overall_accuracy=acc["overall_accuracy"],
             cum_convW_moved_pct=mv, pct_steps_zero_move=100.0 * zero_steps / n_steps, seconds=time.time() - t0)
    mw.log(f"[{tag}] DONE bal_acc={r['balanced_accuracy']:.2f}%  convW moved={mv:.1f}%  zero-move steps={r['pct_steps_zero_move']:.1f}%  ({r['seconds']:.0f}s)")
    return r

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    calib = os.environ.get("CALIB", "absmax"); mw.log(f"[exp14] calibration = {calib}")
    model, params, trX, trY, evX, evY, acts0 = build_folded(calib)
    zs = mw.evaluate(model, evX, evY); mw.log(f"[exp14] FOLDED zero-shot: {zs}")
    regimes = [s for s in os.environ.get("REGIMES", "direct@3e-6,direct@1e-5,direct@3e-5,direct@1e-4,master@3e-6,master@1e-5,master@3e-5").split(",") if s]
    out = {"design": "BN folded into QuantConv2d (weight+bias), BN->Identity", "calibration": calib, "zero_shot": zs,
           "n_trainable": len(params), "kinds": {k: sum(p["kind"] == k for p in params) for k in ("quant", "float")}, "runs": {}}
    for spec in regimes:
        mode, lr = spec.split("@"); lr = float(lr); t0 = time.time()
        if mode.endswith("_lsb"): r = run_regime_lsb(model, params, trX, trY, evX, evY, mode, lr, float(os.environ.get("EPS_LSB", "1")), tag=spec)
        else: r = mw.run_regime(model, params, trX, trY, evX, evY, mode, lr, tag=spec)
        r["seconds"] = time.time() - t0; out["runs"][spec] = r
        mw.log(f"[exp14] {spec}: bal_acc={r.get('balanced_accuracy', float('nan')):.2f}%  ({r['seconds']:.0f}s)")
        with open(os.path.join(HERE, os.environ.get("RESULTS", "results.json")), "w") as f: json.dump(out, f, indent=2, default=float)
    mw.log("[exp14] DONE")

if __name__ == "__main__": main()
