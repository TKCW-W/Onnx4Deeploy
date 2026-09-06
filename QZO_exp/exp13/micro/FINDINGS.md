# exp13 micro-trace — where does the device/host mismatch first appear? (2026-09-07 01:45 CEST)

Setting: exp13 (lr 3e-6, conv weight+bias frozen → constant), step-0 +eps forward, training window 0.
Method: the step-0 forward rebuilt as an inference graph (fp32 params perturbed exactly as the train graph does;
raw biases injected so host and device each add one rounding constant) — verified **bit-identical to the frozen train
graph on the host for all 35 tensors**. Then `make_probe13.py` truncates it at each fp32 point, the device runs it
untiled with the harness compare tolerance set to 0 (`OUTPUT_TOL=0.0f`, `OUTPUT_BITS=ON` → raw bits of every
non-identical element). `Dequant-k` probes the conv-k requant int8 output exactly (dequant is an exact per-element scale).

## Result (probe_summary.txt)
| tensor | non-identical | mostly | max ulp |
|---|---|---|---|
| Dequant-0 (≡ input Quant → conv-0 → RequantShift-0) | **0 / 78,512** | — | 0 |
| BN-0 | 19,841 / 78,512 (25%) | 1 ulp (80%) | 4 |
| MaxPool-0 | 2,543 / 9,744 | 1 ulp | 2 |
| Dequant-1 | **0 / 19,712** | — | 0 |
| BN-1 | 7,916 / 19,712 (40%) | 1 ulp | 10 |
| MaxPool-1 | 1,163 / 4,928 | 1 ulp | 8 |
| Dequant-2 | **0 / 5,152** | — | 0 |
| BN-2 | 2,139 / 5,152 (42%) | 1 ulp | 3 |
| MaxPool-2 | 348 / 1,120 | 1 ulp | 3 |
| Dequant-3 | **0 / 1,280** | — | 0 |
| BN-3 | 563 / 1,280 (44%) | 1 ulp | 12 |
| Dequant-4 | **0 / 320** | — | 0 |
| BN-4 | 146 / 320 (46%) | 1 ulp | 32 |
| GAP | 8 / 32 | 1 ulp | 3 |
| logits (fc Gemm) | 7 / 9 | 1–2 ulp | 2 |
| loss (round-1 log, same forward) | `3eb41bc8` vs `3eb41bcd` | | 5 |

## Reading
1. **The integer path is bit-exact at every block** (input Quant, int8 conv, RequantShift rounding, Dequant): 0 differing
   elements in all five conv+requant outputs.
2. **The first difference appears at BN-0**, the first fp32 kernel, at 1–4 ulp on ~25% of elements; every BN re-creates
   it (25–46%). Larger ulp counts (≤32) occur on near-zero BN outputs where `(x−mean)·inv_std·g + b` cancels — still
   ≤5e-7 absolute. This matches the disassembly of `BatchNorm.c.obj` (13 fused `fmadd/fmsub.s`, reassociation under
   `-ffast-math`) vs the host's source-order mirror.
3. **The next Quant absorbs it completely** at step 0 (no int8 boundary crossed on this window) — the difference does not
   accumulate through the blocks; it is re-seeded by each BN and only BN-4's noise (no Quant after it) reaches the loss via
   GAP (8/32) and the fc (7/9 logits, 1–2 ulp; the fc Gemm adds its own FMA contribution) → loss 5 ulp.
4. Over a carry: the ulp noise perturbs `g` → fp32 parameter drift (exp12/13); when the drifted BN output lands on an int8
   boundary the Quant stops absorbing → LARGE events. The int path never participates at step 0.

## Causal confirmation proposed (not run — awaiting go/no-go)
Rebuild ONLY `TargetLibraries/PULPOpen/src/BatchNorm.c` with `-fno-fast-math -ffp-contract=off` (the one combination
measured to give fused=0) and re-run probe 1: expected **0 / 78,512**. One kernel, one tensor.

## Side finding (to re-check later)
The exp9/exp10 accuracy-eval fixture injected `bias_rqsadd` WITH the baked div/2 while the host initializer path and the
device merge pass each add another div/2 → every conv output rounded UP instead of to-nearest (+0.5 LSB systematic;
54% of requant outputs shifted by 1 LSB on window 0). Device and host were consistent with each other, but the eval
forward was not the training forward. Fix = inject `bias_rqsadd − div/2` (done here via `step0_peps_params_rawbias.npz`).

Files: `make_step0_dump.py`, `_rebuild_fixed.py`, `_host_layerwise.py`, `make_probe13.py`, `run_probe_sweep.sh`,
`analyze_probes.py`, `probe_results.txt`, `probe_summary.txt`, `qzo13_probe_NN.log.gz`, `qinfer_step0_fixed/`.
