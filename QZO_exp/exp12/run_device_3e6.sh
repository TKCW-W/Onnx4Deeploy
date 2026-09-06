#!/bin/bash
# exp12 full device round-1 at lr 3e-6. Run on the HOST; drives the traindeeploy container.
# Precondition: /app/Onnx4Deeploy/QZO_exp/exp12/baked_3e6/outputs.npz exists (3e-6 export finished).
set -e
pgrep -f "[g]vsoc_launcher" | xargs kill -9 2>/dev/null || true; echo "orphans: $(pgrep -f '[g]vsoc_launcher' | wc -l)"
docker exec traindeeploy bash -c '
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 experiments/zo_smoke/pack_2step_fixture.py /app/ETH/Onnx4Deeploy/QZO_exp/exp12/baked_3e6 /app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet speechnet_qzo_lr3e6_train speechnet_qzo_lr3e6_update'
docker exec -d traindeeploy bash -c 'cd /app/ETH/TrainDeeploy/DeeployTest && python3 deeployMezoRunner_tiled_siracusa.py -t Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_train --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo_lr3e6_update --n-steps 2700 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 --l1 128000 --l2 2000000 --cores 8 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON > /app/ETH/Onnx4Deeploy/QZO_exp/exp12/device_round1_3e6.log 2>&1'
echo "launched full 3e-6 device round-1 -> exp12/device_round1_3e6.log"
