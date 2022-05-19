# Pre-training

In addition to the scripts within this directory, [HuggingFace's `transformers` script](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py) is also referenced.
*It's ideal that the library version matches when doing task fine-tuning.*
If training was interrupted, it can be resumed from the most recent checkpoint by running the script in the same way & specifying `--resume_from_checkpoint` as an additional argument (specifying the most recent model checkpoint).

The steps below outline the options used to pre-train the models.
Especially if you are using the same batch size, you'll probably need to run this on [multiple GPUs](https://huggingface.co/docs/transformers/v4.16.2/en/performance#multigpu-connectivity), modifying the `per_device_train_batch_size` & `per_device_eval_batch_size` accordingly.

After following a set of the below steps, the resulting pre-trained model will be located in the directory referenced as `$MODEL_PATH`.


## BERTu: BERT trained on the target language from scratch

1. Load the BERT model & tokeniser configurations:
   ```shell
   python load_configuration.py \
       --name=bert \
       --path=$MODEL_PATH \
       --tokeniser_data=$UNLABELLED_TRAINING_DATA_PATH \
       --vocabulary_size=52000
   ```
   *If need be, modify the `config.json` (e.g. reducing `num_hidden_layers`).*
3. Execute the pre-training script:
   ```shell
   python transformers/examples/pytorch/language-modeling/run_mlm.py \
       --model_type=bert \
       --config_name=$MODEL_PATH \
       --tokenizer_name=$MODEL_PATH \
       --dataset_name=MLRS/korpus_malti \
       --dataset_config_name=shuffled \
       --do_train \
       --do_eval \
       --per_device_train_batch_size=512 \
       --per_device_eval_batch_size=256 \
       --evaluation_strategy=epoch \
       --max_steps=9000000 \
       --warmup_steps=10000 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --max_seq_length=128 \
       --output_dir=$MODEL_PATH \
       --save_strategy=epoch \
       --overwrite_output_dir \
       --no_log_on_each_node \
       --fp16 \
       --preprocessing_num_workers=16
   ```
4. Re-execute the pre-training script, modifying the following arguments:
   - Add `--model_name_or_path=$MODEL_PATH`
   - Remove `--config_name` & `--tokenizer_name`
   - Change `--max_seq_length=512`
   - Change `--max_steps=100000`

## mBERTu: Multilingual BERT adapted to the target language

1. Generate the pre-trained multilingual BERT model & tokeniser:
   ```shell
   python load_configuration.py \
       --name=bert-base-multilingual-cased \
       --path=$MODEL_PATH \
       --tokeniser_data $UNLABELLED_TRAINING_DATA_PATH $UNLABELLED_EVALUATION_DATA_PATH \
       --vocabulary_size=5000
   ```
   *If you don't want to augment the pre-trained model's vocabulary omit the `tokeniser_data` & `vocabulary_size` arguments.*
2. Execute the pre-training script:
   ```shell
   python transformers/examples/pytorch/language-modeling/run_mlm.py \
       --model_type=bert \
       --model_name_or_path=$MODEL_PATH \
       --dataset_name=MLRS/korpus_malti \
       --dataset_config_name=shuffled \
       --do_train \
       --do_eval \
       --per_device_train_batch_size=512 \
       --per_device_eval_batch_size=256 \
       --evaluation_strategy=epoch \
       --max_steps=9000000 \
       --warmup_steps=10000 \
       --learning_rate=1e-4 \
       --weight_decay=0.01 \
       --max_seq_length=128 \
       --output_dir=$MODEL_PATH \
       --save_strategy=epoch \
       --overwrite_output_dir \
       --no_log_on_each_node \
       --fp16 \
       --preprocessing_num_workers=16
   ```

## mBERT: Multilingual BERT as is

1. Generate the pre-trained model configuration:
   ```shell
   python load_configuration.py \
       --path=$MODEL_PATH \
       --name=bert-base-multilingual-cased
   ```
