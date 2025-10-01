# Fine-tuning

Fine-tuning to a downstream tasks requires a [pre-trained model](../pretrain).
Even if using an off-the-shelf model without any pre-training adjustments, the model should be stored to disk & referenced that way.


## Datasets

The following are the datasets used, & how to access them:

- **[MUDT (UniversalDependencies)](https://github.com/UniversalDependencies/UD_Maltese-MUDT/)**
- **[MLRS POS Gold data](https://mlrs.research.um.edu.mt/)**
  combine the data from the different domains & split them into training, development, & testing files,
  using [the convertor script](../data/mlrs/convert_data.py):
  ```shell
  python mlrs/convert_data.py part_of_speech \
  --path=$LABELLED_DATA_PATH \
  --generated_path="$LABELLED_DATA_PATH/mlrs_pos.tsv" \
  --dev_proportion=0.1 \
  --test_proportion=0.1
  ```
- ~~**[WikiAnn](https://github.com/afshinrahimi/mmner)**~~
  _deprecated in favour of MAPA._
- **[MAPA](huggingface.co/datasets/MLRS/mapa_maltese)**
- **[Sentiment Analysis](https://github.com/jerbarnes/typology_of_crosslingual/tree/master/data/sentiment/mt)**


## Training

> [!NOTE]
> The code available here is compatible with [Hugging Face `transformers`](https://github.com/huggingface/transformers/) and is the **recommended version to use**.
> An [earlier version](https://github.com/MLRS/BERTu/tree/2022.deeplo-1.10/finetune), based on [AllenNLP](https://github.com/allenai/allennlp), is available to allow for the replication of the original results.

Run a fine-tuning run by executing the following command:
```shell
python run_${task}.py \
   $DATASET_ARGS \
   --model_name_or_path=MLRS/BERTu --classifier_dropout=0.1 --seed=$seed \
   --do_train --num_train_epochs=200 --early_stopping_patience=20 --per_device_train_batch_size=16 --learning_rate=2e-5 --lr_scheduler_type=inverse_sqrt --warmup_ratio=0.005 --weight_decay=0.01 \
   --do_eval --eval_strategy=epoch --per_device_eval_batch_size=32 $METRIC_ARGS --load_best_model_at_end --greater_is_better=true \
   --do_predict \
   --logging_strategy=epoch --logging_steps=1 --logging_first_step=true \
   --save_strategy=epoch --save_total_limit=20 --output_dir=$OUTPUT_PATH
```
The values used for `$seed` are `0`, `1`, `2`, `3`, & `4`.
Task-specific values are summarised below:


| Task Name | `${task}`      | `$DATASET_ARGS`                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `$METRIC_ARGS`                                                                               |
|-----------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| MLRS POS  | tagging        | `--dataset_name="mlrs_pos.py" --train_file="$DATA_PATH/mlrs_pos/train.tsv" --validation_file="$DATA_PATH/mlrs_pos/dev.tsv" --test_file="$DATA_PATH/mlrs_pos/test.tsv" --text_column_name="tokens" --label_column_names="xpos"`                                                                                                                                                                                                                                                        | `--metric_name="poseval" --metric_for_best_model="accuracy"`                                 |
| MAPA      | tagging        | `--dataset_name="MLRS/mapa_maltese" --text_column_name="tokens" --label_column_names="level1_tags"`                                                                                                                                                                                                                                                                                                                                                                                   | `--metric_name="seqeval" --metric_for_best_model="overall_f1"`                               |
| Sentiment | classification | `--train_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/train.csv" --validation_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/dev.csv" --test_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/test.csv"  --dataset_kwargs="{\"names\": [\"label\", \"text\"]}" --text_column_names="text" --label_column_name="label"` | `--metric_name="f1" --metric_kwargs="{\"average\": \"macro\"}" --metric_for_best_model="f1"` |


## Acknowledgements

Some of this code was adapted from [MELABench](https://github.com/MLRS/MELABench/tree/main/finetuning), which has information about other tasks not included here.
