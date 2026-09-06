# exp13 — control: train ONLY the fp32 parameters (conv weight + bias frozen), lr 3e-6

Started 2026-09-06 21:25 CEST · follows exp12 (4,254/21,600 errors at 3e-6 with int8 weights frozen but int32
biases still training). Question: how much of exp12's residual is the bias int path vs the float path?

## Mechanism (no graph-structure change)
Freeze = zero every `*_pmul` initializer consumed by an `RQSPerturbRademacher` node in BOTH graphs
(`qzo_transform.freeze_conv_pmul`, env `QZO_FREEZE_CONV=1` in the exporter). With `mul == 0` the RQS perturbation is
exactly `(rad*0 + 2^(S-1)) >> S == 0` for the +-eps forward and for the update, on device (compiled-in constant) and
host (initializer) alike; the host sim additionally skips the rqs update branch. `inputs.npz`, node ids and the
float-parameter noise `z` are identical to exp12, so exp13 differs from exp12 ONLY by the freeze. lr 3e-6 as exp12.

## Reading
| errors vs exp12's 4,254 | meaning |
|---|---|
| ≪ 4,254 | the int32 bias path was a major amplifier of the float drift |
| ≈ 4,254 | the residual is the float path alone (fp32 param drift + fp32 forward) |

## Commands (as executed; overlapped)
1. Device fixture: copy `exp12/baked_3e6` → `exp13/baked_3e6_freeze`, apply `freeze_conv_pmul` to both graphs (in
   `agitated_hugle`), pack with `pack_2step_fixture.py ... speechnet_qzo_lr3e6_freeze_train speechnet_qzo_lr3e6_freeze_update`,
   run `deeployMezoRunner_tiled_siracusa.py ... --n-steps 2700 --lr 3e-6 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`
   → `device_round1_3e6_freeze.log` (compiled-in reference = exp12's placeholder; its printed `Errors:` line is void).
2. Host reference: exp12's export command with `QZO_FREEZE_CONV=1`, `-o exp13/baked_3e6_freeze_ref` → `export_3e6_freeze.log`.
3. Checks: export log shows `[QZO_FREEZE_CONV] zeroed 10 *_pmul initializers` for both graphs; the export's graphs must be
   byte-identical to the hand-frozen device fixture; device step-0 `lp_bits` == export step-0 L+ (bit) — differs from
   exp12's step-0 (no conv perturbation) by design.
4. Analysis: `analyze_exp12.py <device log> <ref outputs.npz>`-style recount (harness rule) + final-param bit-compare:
   int8 weights AND int32 biases must be bit-exact; only fp32 params may differ.
