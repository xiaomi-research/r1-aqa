#!/bin/bash


TEST_FILE=data/MMAU/mmau-mini.data

iters=(100 200 300 400 500 600 700 800 900 1000)
for iter in ${iters[*]}; do
    MODEL_DIR=exp/model/checkpoint-${iter}
    OUT_DIR=exp/model/test_${iter}
    mkdir -p ${OUT_DIR} || exit 1

    python src/test.py --model_path ${MODEL_DIR} --data_file ${TEST_FILE} --out_file ${OUT_DIR}/res_mmau.json || exit 1
    python data/MMAU/evaluation.py --input ${OUT_DIR}/res_mmau.json > ${OUT_DIR}/eval_mmau.json || exit 1
done

# show Acc for each checkpoint
python src/utils/show_acc.py -i exp/model || exit 1
