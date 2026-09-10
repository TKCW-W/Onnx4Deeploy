# exp15 — ECO for on-device quantized ZO fine-tuning

Transfer of **ECO** (arXiv:2601.22101, "Quantized Training without Full-Precision Master
Weights") to our SpeechNet INT8 zeroth-order (MeZO/SPSA) fine-tuning. Paper summary +
proposed transfer: `../../../docs/eco_paper_summary.md` (§8 is the transfer, not a paper result).

## The problem ECO addresses (= our LSB stall)

Applying an optimizer update straight to quantized weights discards every update smaller than
0.5 LSB. In our setting, at the natural small learning rate the direct-INT8 ZO update
**stalls completely**:

| regime (pooled@99.99, 300-step, from exp_calibration) | conv weights moved | acc |
|---|---|---|
| direct-INT8 ZO, lr 3e-6 | **0.00%** (100% of steps move nothing) | 85.56% = zero-shot |
| master-weight ZO, lr 3e-6 | 50.6% | 85.00% |
| direct-INT8 ZO, lr 1e-5 (the working knob) | 65.9% (round 1) | 90.00% |

The master-weight version learns but costs a full **FP32 shadow buffer (~62 KB L2)**. ECO's
claim: recover the master-weight trajectory by injecting the per-step rounding residual into
the **SGD momentum buffer** — momentum doubles as the error-feedback accumulator, so there is
no master copy and no separate error buffer.

## Method (mapped to our setting)

Per quantized weight, in the dequantized/real domain (`s_w` = frozen per-channel scale):
```
g        = (Lp − Lm)/(2·eps·n_accum) · z          # ZO gradient estimate
m_tilde  = beta·m + (1−beta)·g                     # SGDM first moment
th_tilde = w_int·s_w − lr·m_tilde                  # tentative real step
w_new    = clamp(quant(th_tilde/s_w), −127, 127)   # RTN or stochastic rounding
e        = th_tilde − w_new·s_w                     # rounding residual
m        = m_tilde + (1/lr)(1 − 1/beta)·e           # ECO injection (memory-free, Alg. 2)
```
`α = (1/lr)(1 − 1/beta)` is **pinned** by lr and beta (Lemma 3.5 virtual-sequence cancellation)
— not a tunable. Exact-EF variant (§2.2) instead stores `e_t`:
`m = m_tilde + (1/lr)·e_prev − (1/(lr·beta))·e`. Float params (BN γ/β) use plain SGDM.

Intuition for why it breaks the stall: when no weight moves, `w_new = w_int` so `e = −lr·m_tilde`
and the injection gives `m ← m_tilde/beta`, i.e. momentum **accumulates `(1−beta)/beta·g` per
step**. The discarded sub-LSB updates are not lost — they build in `m` until `lr·m` crosses
0.5 LSB and the integer weight steps. That is error feedback, paid for with a buffer SGDM
already has.

## Arms (`run_eco.py`, reusing exp_calibration `run_study`/`run_incremental`)

Same faithful Brevitas SpeechNet, frozen pooled@99.99 act scales + frozen pretrained
per-channel weight scales, incremental 4-round protocol (ft batch r → eval batch r+1), z-seed
restart per round, n_accum 4, eps 0.01.

| arm | rule | lr | β |
|---|---|---|---|
| `direct_3e6` | direct INT8 RTN (baseline) | 3e-6 | — |
| `direct_1e5` | direct INT8 RTN (working ref ~86%) | 1e-5 | — |
| `eco_mf_3e6_b90` | ECO memory-free, RTN | 3e-6 | 0.90 |
| `eco_mf_3e6_b99` | ECO memory-free, RTN | 3e-6 | 0.99 |
| `eco_mf_3e6_b90_sr` | ECO memory-free, **SR** | 3e-6 | 0.90 |
| `eco_ex_3e6_b90` | exact EF (stores e) | 3e-6 | 0.90 |
| `sgdm_none_3e6_b90` | momentum, **no EF** (ablation) | 3e-6 | 0.90 |

The ablation isolates whether momentum alone (no residual injection) breaks the stall — it
should not, since rounding still discards sub-LSB steps.

## Metrics logged per round (paper §8.6)

- Post-adaptation balanced accuracy (vs the ~86% direct@1e-5 / float-ZO / master anchors).
- % of steps with any conv movement; % conv weights moved (this round / cum vs pretrained / ever-union).
- Steps-to-first-move distribution (median, p90) — the direct stall measurement.
- **cos(e_t, e_{t+1})** and **‖e_{t+1}‖/‖e_t‖** — the ECO go/no-go: high cos ⇒ memory-free viable;
  near-zero ⇒ fall back to exact EF.

## Deliverables

`Findings.md` with the arm comparison (accuracy + movement + cos(e)), the go/no-go verdict,
and a section on the **hardware / on-device-simulation implementation plan** (kernel, buffers,
fixed-point range for α, integration into the QZO update graph).
