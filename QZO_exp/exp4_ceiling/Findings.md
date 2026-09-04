# exp4_ceiling — abs-ceiling update instead of rounding — Findings

Date: **2026-09-04** · `run_ceiling.py`, `results.json`, `run.log` · Setting: float-ZO lr 3e-6,
round 1 (ft b1 → eval b2, 2700 steps, ε 0.01, seed 42), pooled@99.99 + abs-max scales frozen,
same harness as the exp_calibration rows it compares against.

## Proposal under test

Replace the update rounding with an absolute ceiling so every nonzero coefficient moves the
weight at least one LSB:

```
u = coeff·z/s_w[c];   delta = sign(u)·ceil(|u|)     # instead of round(u)
```

Goal (supervisor): push more steps to update the conv weights at the float-ZO lr, without the
memory cost of master weights.

## Result — un-stalls completely, and destroys accuracy

| metric | value |
|---|---|
| steps that move conv weights | **2700 / 2700 (100%)** |
| conv weights moved per step | **100.0%** (all 14,880, every step) |
| cumulative: net / union | 98.4% / 100% |
| zero-shot b2 | 85.56% |
| b2 during training | 72.22 → 60.00 → 59.44 → **51.67%** (monotone decay) |
| **final b2** | **51.67% — 34 pts BELOW zero-shot** |

Comparison on the identical harness:

| update rule @ lr | final b2 | conv moved |
|---|---|---|
| round @ 3e-6 (stall) | 87.78% | 0% |
| round @ 1e-5 (tail filter) | 90.00% | 65.9% net |
| master @ 3e-6 | 87.2–88.3% | ~76% |
| stochastic rounding | degraded (rejected earlier) | moves |
| **ceil @ 3e-6** | **51.67%** | 100%/step |

## Why (mechanism)

At lr 3e-6 the intended per-step update is ~0.03 LSB. `ceil` inflates every such update to a
full ±1 LSB — a **~33× step amplification applied indiscriminately**: the per-element update
becomes `−sign(g)·z · 1 LSB`, i.e. sign-SGD on the ZO estimate at a fixed 1-LSB step, where the
per-step, per-element signal (one scalar sign × a random ±1 direction) is almost pure noise.
Every weight random-walks away from the pretrained solution at ~1 LSB/step; the monotone
accuracy decay is that walk in action. This is the over-movement regime from the lr sweep
(round@6e-5 → 70.6%, @1e-4 → 22.8%) taken to its limit — reached from below, at the "safe" lr.

Contrast with the two mechanisms that DO work, which are both selective/faithful where ceil is
neither:
- **round @ tuned lr** keeps rounding's threshold but places it at the |g| tail → only
  strong-signal steps fire (34 of 2700), and those carry real gradient information.
- **master weights / error feedback** keep the true sub-LSB magnitudes and let them accumulate
  → a weight moves only when the *summed* evidence crosses ½ LSB.
`ceil` does the opposite of both: it moves on every step (no selectivity) by ≥1 LSB regardless
of the evidence (no magnitude fidelity). Note stochastic rounding — the *unbiased* memoryless
version of the same wish (E[δ]=u) — already degraded accuracy in our earlier test; ceiling is
strictly more biased (E[δ] ≈ sign(u)·1 for sub-LSB u) and correspondingly worse.

## Verdict

Abs-ceiling is refuted as a stall remedy: it converts the stall into aggressive noise injection
and costs 34 accuracy points at the setting proposed. The un-stall levers remain lr placement
(validated, fold-tuned) and accumulation (master / error feedback) — both preserve the property
that movement reflects gradient evidence, which ceiling removes.
