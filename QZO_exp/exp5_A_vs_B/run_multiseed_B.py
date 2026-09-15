#!/bin/sh
# SPDX-License-Identifier: MIT
# exp5 step 1 — multi-seed on the TRUE INTEGER datapath (B-host).
#
# Question: is B's -1.11 (85.00 -> 83.89) systematic, or one unlucky z-draw? B has been run
# exactly once (seed 42). Here: 3 fresh seeds at 100 epochs (1350 steps, per exp18 the knee of
# the epoch/benefit curve), same fold-3 protocol, pooled@99.99 baked, lr 1e-5.
#
# Each seed needs its own export (the Rademacher seed is baked into the graph attributes);
# the seed-patching sim (~12 steps/min) makes each ~2h. Run sequentially in agitated_hugle.
#
#   nohup sh run_multiseed_B.py > multiseed.log 2>&1 &
set -e
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
TH=/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1/fixture/pooled_9999_fold3.json
cd /app/Onnx4Deeploy
for SEED in 1 7 123; do
  OUT=/app/Onnx4Deeploy/QZO_exp/exp5_A_vs_B/B_seed${SEED}
  echo "=== [B multiseed] seed ${SEED} -> ${OUT} ==="
  rm -rf "$OUT"
  QZO_POOLED_THRESHOLDS=$TH PYTHONDONTWRITEBYTECODE=1 \
  python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train \
    --noise-type rqs_rademacher \
    --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
    --pretrained-weights $CKPT \
    --subject S01 --session 3 --condition vocalized --batch 1 \
    --data-size 54 --stratified --n-epochs 100 --n-accum 4 --lr 1e-5 --seed ${SEED} \
    -o "$OUT" 2>&1 | grep -E "export complete|🧊|Error|Traceback" || true
  echo "=== [B multiseed] seed ${SEED}: evaluating endpoint on batch 2 ==="
  cd /app/TrainDeeploy/DeeployTest/experiments/deliverable/exp9_QZO_round1
  python3 - "$OUT" <<'PY' 2>&1 | grep -E "seed-endpoint|Error"
import sys, numpy as np
o = np.load(sys.argv[1] + "/outputs.npz")
np.savez("/tmp/_ms_w.npz", **{k[len("updated_"):]: o[k] for k in o.files if k.startswith("updated_")})
PY
  python3 build_qzo_infer_fixture.py --fixture-dir "$OUT" \
    --dump-npz /tmp/_ms_w.npz --out-dir /tmp/_ms_fix_${SEED} 2>&1 | grep -oE "balanced=[0-9.]+%" \
    | sed "s/^/  seed ${SEED} endpoint b2: /"
  cd /app/Onnx4Deeploy
done
echo "==== B multiseed complete ===="
