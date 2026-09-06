# exp11 — Findings (interim, 2026-09-06 17:30 CEST)

Branch `feat/QZO`. Deliverable: device ↔ host-reference bit-exactness (under the float-ZO tolerance) over a full
round-1. **Status: not yet achieved.** What was accepted, what was measured, what was refuted, and what is pending.

## 1. Accepted changes — host reference made faithful to the device, op for op (`onnx_node_implementations.py`)
All originals kept commented (`# [exp11-orig]`). Each mirrors the device kernel/template bit-for-bit in fp32.

**Quant** (L1.1) — shipped `QuantTemplate` MULTIPLIES by `scale`; our exporter emits the ONNX step size, so the device
parser passes the fp64 reciprocal, cast to fp32. The host DIVIDED in fp64 (a different op in a different domain):
```python
_inv = np.float32(1.0 / float(attrs["scale"]))      # == device (float32_t)(1.0/float(node.attrs['scale']))
_scaled = x.astype(np.float32) * _inv                # fp32 mul
_shifted = _scaled + _zp                             # zp BEFORE rounding, as the template
_q = np.trunc(_shifted + np.where(_shifted >= 0, np.float32(0.5), np.float32(-0.5)).astype(np.float32))
```
**Dequant** (L1.1): fp32 `((float32)q - (float32)zp) * (float32)scale` (was fp64 then cast — double rounding).
**BatchNormInternal** (L1.2) — the BP-inherited frozen kernel (NOT the shipped inference BN, which we do not use):
```python
_inv_std = np.float32(1.0) / np.sqrt(_vr + _eps32)   # device: 1.0f / sqrtf(running_var + eps), per channel
y = (((_x32 - _mn) * _inv_std) * _sc + _bs)          # device: (x - mean) * inv_std * g + b (C left-assoc)
```
**GlobalAveragePool** (L2.1): sequential `acc = acc + x[..., i]` in memory order, then `* (1.0f/HW)` (was `np.mean`).
**Gemm** (L2.2, `!transA && transB`, M=1,N=32,O=9): per output, for k<N_unroll step 6 add the 6-term left-assoc
temporary `((((p0+p1)+p2)+p3)+p4)+p5` to the accumulator, singles for the tail, then `+ bias` (was BLAS matmul).
**SoftmaxCrossEntropyLoss** (L2.3): sequential `sum += exp(d_j)` over classes, `lp = (logit - max) - log(sum)`.

## 2. Step-0 measurement per level (device bits `3f9d48ea 3e2d5aa0 3dffdca0 3fc556fd`; harness `check_step0.py`)
| level | a0 | a1 | a2 | a3 | exact |
|---|---|---|---|---|---|
| baseline | 3f9d48e4 | 3e2d5aa1 | 3dffdc94 | 3fc556ff | 0/4 |
| L1.1 Quant/Dequant | (unchanged) | | | | 0/4 |
| L1.2 BN | 3f9d48e5 | 3e2d5aa1 | 3dffdc94 | 3fc556fe | 0/4 |
| L2.1 GAP | 3f9d48e8 | 3e2d5aa1 | 3dffdc94 | 3fc55700 | 0/4 |
| L2.2 Gemm | 3f9d48e6 | 3e2d5aa1 | 3dffdc94 | 3fc556ff | 0/4 |
| L2.3 SCE | 3f9d48e6 | 3e2d5aa1 | 3dffdc94 | 3fc55700 | 0/4 |
Samples a1, a2 never moved through all six alignments → the residual is unreachable by source-order mirroring.

## 3. Root cause, reviewed twice
**(a) The forward residual is device-side compiler/libm, not our code.** Disassembly of the ACTUAL round-1 build objects
(`fma_scan.py`): `Gemm.c.obj` **154 fused `fmadd/fmsub.s`**, `BatchNorm.c.obj` **13**, `GlobalAveragePool.c.obj` 0; the SCE
calls picolibc `expf`/`logf`. Only `-fno-fast-math -ffp-contract=off` together yields fused=0 (`-fno-fast-math` alone
leaves 133; pragmas do nothing under the global `-ffast-math`). **Not applied** — paused per the user until exp12.

**(b) The update path is identical, device vs host, in source and compiled form** (`NOTES.md` table): `acc += (lp-lm)`,
`(2·eps)·n_accum`, `acc/denom`, `(−lr)·g_proj`, `override/eps_baked`, `lrintf(m·eps_scale)` ≡ `np.rint`, fc `w ± coeff`;
the two divisions compile to `fdiv.s` (`fdiv_scan.py`). The update does NOT compute a different reference loss.

**(c) What the 1e-5 divergence actually is** (device `lp_bits`/`lm_bits` vs host, `analyze_exp12.py`, recount == harness
`Errors: 16565 out of 21600` exactly): L+ first LARGE at step 5 — *transient*, steps 25–300 return to 98–100% ulp-only, so
the weights were still identical; L− first LARGE at **step 326**; both 67% LARGE in 300–600 and 100% after 600. So the
carry stays identical for ~300 steps with occasional single-forward LARGE events (int8 *activation* flips when a
pre-Quant value sits at a .5 boundary and the device fp32 differs by a ulp), and the weight trajectories bifurcate near
step 326. The L1+L2-aligned host reproduces this timeline **identically** to the original host.

## 4. Surrogate: direct-int8 vs master-weight update under an identical ~3e-7 loss perturbation (`master_surrogate.py`)
Faithful replica of the sim loop (step-0 bits reproduced exactly; RQS delta reproduced exactly), two trajectories per
mode differing only by a deterministic ±3e-7 relative nudge on every L+/L− (3–5 ulp, the measured device-host magnitude),
lr 1e-5, 600 steps, identical `z`:
| update | int8 weights differing at u=100 / 200 / 325 | losses LARGE by 200–300 |
|---|---|---|
| direct-int8 | 0 / 0 / 0 (all 14,984 identical through 600 steps) | 0% |
| master (fp32, relative) | 0 / 2025 / 3059 | 100% |
Reading: the direct update's double integer rounding (`rint`, then shift) is robust to a ulp seed; the master's
continuously drifting values keep many weights near read-out boundaries, so the int8 read-out flickers often and the
feedback compounds. **Whether this refutes master weights for fidelity is an OPEN question the user has flagged** (it
depends on lr — master's operating point was 3e-6, not the 1e-5 used here — and on the surrogate's fidelity to a real
device implementation); to be discussed after exp12. Not a conclusion.

## 5. Pending — exp12 control (running)
Device round-1 at lr 3e-6 (int8 update inert) vs its host reference; `Errors:` count compared with 16565/21600.
Prediction stated before the run: the count will not change much (float part). See `exp12/Plan.md`.

## 6. Plan for item 2 — a faithful PyTorch simulation to retune the on-device QZO setting
The Brevitas fake-quant sim (actor A) is ~2 pt optimistic and diverges from the device at zero-shot (85.00 vs 83.33) because
it rescales by the exact float scale in float, while the device uses a dyadic `mul>>shift` with round-half-up and a
fixed-point requant (exp10 §three-way). Two routes, both keeping the user's rules (SpeechNet, no toy, shipped-faithful):
1. **Use B-host as the simulator.** `run_onnx_graph` is now op-for-op faithful to the device (this exp); it already runs
   the full carry (~4 s/step). Retune lr / eps / n_accum / calibration on it directly; it IS the deployable datapath.
   Cost: slow for sweeps; mitigate by sweeping on shorter horizons + multi-seed, confirming winners on the device.
2. **Make the PyTorch sim device-faithful** (fast sweeps): replace Brevitas' float rescale with the exported dyadic
   `mul/div` per channel, round-half-up via the baked `div/2`, integer accumulation, and the fp32 tail order — i.e. port
   the six mirrored ops into torch. Validate by matching B-host's zero-shot 83.33% and step-0 losses to ulp; then sweep.
Recommended: (2) for the sweep, (1) + a device smoke (first 100 steps, `Errors:` count) for each candidate, full device
round-1 for the final setting. Success = the sim's accuracy ranking of settings matches the device's.
