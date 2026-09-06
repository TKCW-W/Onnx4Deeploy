# exp11 — evidence log (2026-09-06)

## Harness
`check_step0.py`: host `run_onnx_graph` +eps loss at step 0 (identical weights, seed 42, windows 0..3) vs the device's
logged `lp_bits` (`device_lp_bits.txt`, 10800 = 2700 steps x 4 accum, +eps only). Baseline reproduces exp10 exactly
(0/4 bit-exact, residuals 5.82e-7 / 8.80e-8 / 7.16e-7 / 1.55e-7; params == device fixture).

## Host bits per level (device bits: a0=3f9d48ea a1=3e2d5aa0 a2=3dffdca0 a3=3fc556fd)
| level | change (host-only) | a0 | a1 | a2 | a3 | exact |
|---|---|---|---|---|---|---|
| baseline | — | 3f9d48e4 | 3e2d5aa1 | 3dffdc94 | 3fc556ff | 0/4 |
| L1.1 | Quant fp32 x*fp32(1/s) zp-before-round; Dequant fp32 | 3f9d48e4 | 3e2d5aa1 | 3dffdc94 | 3fc556ff | 0/4 (bit-identical to baseline: absorbed by int8 round; q*13.1875 exact) |
| L1.2 | BN ((x-mean)*inv_std)*g+b (BP-inherited device order) | 3f9d48e5 | 3e2d5aa1 | 3dffdc94 | 3fc556fe | 0/4 (a0,a3 1 ulp closer) |
| L2.1 | GAP sequential sum * (1/HW) | 3f9d48e8 | 3e2d5aa1 | 3dffdc94 | 3fc55700 | 0/4 (a0 +3 ulp closer; a3 overshoots) |
| L2.2 | Gemm !transA&&transB 6-way unroll, 6-term temporaries | 3f9d48e6 | 3e2d5aa1 | 3dffdc94 | 3fc556ff | 0/4 |
| L2.3 | SCE sequential sum | 3f9d48e6 | 3e2d5aa1 | 3dffdc94 | 3fc55700 | 0/4 |
a1, a2 never move through all six alignments -> residual is unreachable by source-order mirroring.

## Why (disassembly of the ACTUAL round-1 build objects, `fma_scan.py`)
Gemm.c.obj fused(fmadd/fmsub.s)=154 separate=28 · BatchNorm.c.obj fused=13 · GlobalAveragePool.c.obj fused=0.
SCE calls picolibc `expf`,`logf` (relocations) under both -ffast-math and -fno-fast-math.
Mechanism (`fma_scan_strict.py`): -fno-fast-math alone -> Gemm still 133 fused (only drops to -ffp-contract=on);
-ffp-contract=off alone -> 133 (global fast-math re-enables); pragmas -> no effect; **-fno-fast-math -ffp-contract=off -> 0**.

## Update path, device vs host (source + compiled) — IDENTICAL
| step | device (`deeploymezotest.c`, templates) | host (`_export_qzo_training`) |
|---|---|---|
| accumulate | `*acc_out += (lp - lm)` | `acc = f32(acc + (f32(Lp) - f32(Lm)))` |
| denom | `2.0f * (float)ZO_EPS * (float)n_accum` -> `(2·eps)·n`, const-folded 0.02f exact, ×4 exact | `f32(f32(2·eps)·n_accum)` same |
| g_proj | `acc / denom` (fdiv.s) | `f32(acc/_denom)` |
| coeff | `-(float)ZO_LR * g_proj` = `(−lr)·g_proj` | `f32(f32(-lr)·g_proj)` |
| eps_scale / ratio | `perturb_eps_override / perturb_eps_baked` (fdiv.s) | `f32(coeff)/f32(eps)` |
| int8 m_val | `lrintf((float)m * eps_scale)` (half-even) | `np.rint(mul.f32 * f32(ratio))` |
| fc float perturb | `w + r*coeff`, r=±1 exact | `w += ±coeff` |
`fdiv_scan.py`: with the exact -ffast-math flags, 2 runtime divisions compile to 2 `fdiv.s` (no reciprocal folding).
=> The update path does NOT compute a different reference loss. The step-0 loss already differs BEFORE any update,
   purely from the forward fp32 tail (Gemm/BN FMA + picolibc expf/logf). The update is an identical amplifier on
   both sides (a ~1e-7 seed flips an LSB only when a channel's m·eps_scale sits near .5 — rare, then cascades).
