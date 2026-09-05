# exp10 — QZO device vs host-ref (run_onnx_graph) divergence: cause + alignment plan

## Classification (from the fixed round-1 run)

The divergence is **fp32 reduction-order non-associativity**, NOT a mechanism/formula difference:
- Step-0 losses agree to 6 decimals: device L+[0]=1.228788 vs ref 1.228787 (diff 1e-6); most pairs
  diff=0. A different algorithm would give a large gap, not last-digit.
- z (noise vector) is IDENTICAL: the 2-step device run was bit-exact 0/16, which is impossible
  unless z and the integer path match exactly. So the RNG/perturbation is NOT the cause.
- The integer path (int8 conv, requant) is bit-exact (same graph). Only the **fp32 tail** differs.

## The exact alignable differences (host `run_onnx_graph` vs device C kernels)

| op | device kernel | host run_onnx_graph | difference |
|---|---|---|---|
| GlobalAveragePool | `GlobalAveragePool.c`: `float sum=0; for i: sum+=in[i]; out=sum*(1/HW)` (SEQUENTIAL, ×inv_HW) | `np.mean(x, axis)` (PAIRWISE sum, ÷HW) | summation order + ×inv vs ÷ |
| fc (Gemm) | pulp matmul kernel (sequential MAC order) | `np.matmul` (BLAS blocked/pairwise) | matmul reduction order |
| SoftmaxCrossEntropyLoss | device fp32 SCE kernel (order + stabilization TBD) | `logits-xm-log(sum(exp))` numpy | reduction order + max-subtract |

All three are IEEE fp32 but accumulate in DIFFERENT ORDER → ~1e-6 last-digit diffs. These are
"by nature" (fp32 non-associativity) but ALIGNABLE: two IEEE-fp32 impls that sum in the SAME
order produce bit-identical results. Amplified by the update `round(coeff·z/s_w)` over the carry
(same round() as the LSB stall) → the growing drift.

## Alignment plan (make host-ref == device bit-exactly)

Reimplement the host `run_onnx_graph` reductions to match the device C kernels EXACTLY, in
np.float32 (not fp64):
1. GlobalAveragePool: sequential `sum += x[i]` in np.float32, then `* np.float32(1/HW)`.
2. Gemm (fc): sequential MAC in the device kernel's index order, np.float32 accumulator.
3. SoftmaxCrossEntropyLoss: match device's max-subtract + sequential exp-sum + log, np.float32.
Gate behind an env flag (QZO_DEVICE_FP32_ORDER) so the default numpy path is preserved.
Verify: recompute the reference step-0..N losses with alignment ON; they must match the DEVICE's
logged losses bit-exactly (diff=0). Then a full re-run should stay bit-exact across all 2700.

## Status
- Device eval (accuracy) still collecting (~126/180 when noted).
- Device fp32 SCE kernel exact order: TBD (Softmax.c is the u8 integer softmax, not the fp32 SCE).
- Next: find the fp32 SCE kernel + fc matmul order, implement aligned reductions, verify vs device.
