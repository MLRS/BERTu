#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=8
export LEARNING_RATE=1e-3
export DROPOUT=0.3

#export DATA_PATH="/data"

readonly output_path="./output/finetune"

mkdir -p $output_path

experiment=$1
task=`echo ${experiment} | cut -d"_" -f1`

data_key=""
test_file=""
metric=""
if [ ${task} == "dp" ];
then
    test_file="${DATA_PATH}/MUDT/mt_mudt-ud-test.conllu"
    data_key="ud"
    metric="ud_LAS"
elif [ ${task} == "upos" ];
then
    test_file="${DATA_PATH}/MLRS POS Gold/mlrs_pos-test.tsv"
    data_key="pos"
    metric="pos_accuracy"
elif [ ${task} == "xpos" ];
then
    test_file="${DATA_PATH}/MLRS POS Gold/mlrs_pos-test.tsv"
    data_key="pos"
    metric="pos_accuracy"
elif [ ${task} == "ner" ];
then
    test_file="${DATA_PATH}/WikiAnn/mt/test"
    data_key="ner"
    metric="ner_f1-measure-overall"
elif [ ${task} == "sa" ];
then
    test_file="${DATA_PATH}/Maltese Sentiment/test.csv"
    data_key="sentiment"
    metric="sentiment_fscore"
fi

for seed in $(seq 0 4); do
    export SEED_SET="${seed}"
    echo ${experiment} ${seed}

    touch $output_path/${experiment}.${seed}/THIS_IS_RUNNING.txt
    allennlp train config/autogen/${experiment}.jsonnet \
        -s $output_path/${experiment}.${seed} --include-package modules \
        && touch $output_path/${experiment}.${seed}/THIS_IS_GOOD.txt \
        || touch $output_path/${experiment}.${seed}/THIS_IS_BAD.txt
    rm -f $output_path/${experiment}.${seed}/THIS_IS_RUNNING.txt

    allennlp evaluate "${output_path}/${experiment}.${seed}/model.tar.gz" "$test_file" \
        --data-key ${data_key} \
        --output-file ${output_path}/${experiment}.${seed}/metrics_evaluation.json \
        --extend-namespace head_tags \
        --extend-namespace head_indices \
        --include-package modules \
        --cuda-device 0
done

python summarise_results.py ${metric} \
    --file_pattern=./output/allennlp/${experiment}.*/metrics.json
