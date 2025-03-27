#!/bin/bash

MMAU_DIR=data/MMAU

iters=(100 200 300 400 500 600 700 800 900 1000)
for iter in ${iters[*]}; do
    MODEL_DIR=exp/model/checkpoint-${iter}
    OUT_DIR=exp/model/test_${iter}

    python src/test_mmau.py --model_path ${MODEL_DIR} --data_file ${MMAU_DIR}/mmau-test-mini.json --audio_dir ${MMAU_DIR} --out_file ${OUT_DIR}/res_mmau_mini.json --batch_size 32 || exit 1
    python ${MMAU_DIR}/evaluation.py --input ${OUT_DIR}/res_mmau_mini.json > ${OUT_DIR}/eval_mmau_mini.txt || exit 1
    # python src/test_mmau.py --model_path ${MODEL_DIR} --data_file ${MMAU_DIR}/mmau-test.json --audio_dir ${MMAU_DIR} --out_file ${OUT_DIR}/res_mmau.json --batch_size 32 || exit 1
done

# show Acc for each checkpoint (test-mini)
python src/utils/show_acc.py -i exp/model || exit 1
