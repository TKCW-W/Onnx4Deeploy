#!/bin/bash
# exp13 micro-trace: device probe sweep over the fp32 points of the (fixed) step-0 +eps inference graph.
# Usage: run_probe_sweep.sh "<extra runner flags e.g. --tolerance 0>"   (run on the HOST)
EXTRA="$1"; M=/Users/qiwenwu/ETH/Onnx4Deeploy/QZO_exp/exp13/micro; OUT=$M/probe_results.txt; : > $OUT
NODES=("QCDQ_/blocks_0_conv_output_dequant/Mul_Dequant" "/bn/BatchNormalization" "/pool/MaxPool"
 "QCDQ_/blocks_1_conv_output_dequant/Mul_Dequant" "/bn_1/BatchNormalization" "/pool_1/MaxPool"
 "QCDQ_/blocks_2_conv_output_dequant/Mul_Dequant" "/bn_2/BatchNormalization" "/pool_2/MaxPool"
 "QCDQ_/blocks_3_conv_output_dequant/Mul_Dequant" "/bn_3/BatchNormalization"
 "QCDQ_/blocks_4_conv_output_dequant/Mul_Dequant" "/bn_4/BatchNormalization"
 "/global_pool/GlobalAveragePool" "/wrappedInnerForwardImpl_5/Gemm")
k=0
for node in "${NODES[@]}"; do
  tag=$(printf "qzo13_probe_%02d" $k)
  docker exec agitated_hugle bash -c "cd /app/TrainDeeploy/DeeployTest && PROBE_SRC=/app/Onnx4Deeploy/QZO_exp/exp13/micro/qinfer_step0_fixed/network.onnx PROBE_INP=/app/Onnx4Deeploy/QZO_exp/exp13/micro/probe_input.npz python3 /app/Onnx4Deeploy/QZO_exp/exp13/micro/make_probe13.py --node '$node' --tag $tag" > /dev/null 2>&1
  pgrep -f "[g]vsoc_launcher" | xargs kill -9 2>/dev/null
  docker exec traindeeploy bash -c "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && python3 deeployRunner_siracusa.py -t Tests/Models/$tag --cores 8 -vv -D BN_FROZEN_STATS=ON $EXTRA" > $M/$tag.log 2>&1
  err=$(grep -oE "Errors: [0-9]+ out of [0-9]+" $M/$tag.log | tail -1); mx=$(grep -oE "Diff: *[-0-9.e+]+" $M/$tag.log | sed "s/Diff: *//" | awk '{v=$1<0?-$1:$1; if(v>m)m=v} END{printf "%.3e", m+0}')
  echo "$(date +%H:%M) probe_$k $node | ${err:-NO VERDICT} | max|diff|=$mx" | tee -a $OUT
  k=$((k+1))
done
echo "SWEEP_DONE" | tee -a $OUT
