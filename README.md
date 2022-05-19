# BERTu: A BERT-based language model for the Maltese language :malta:

This repository contains code & information relevant for the paper **Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and BERT Models for Maltese**.

The pre-trained language models can be accessed through the Hugging Face Hub using [`MLRS/BERTu`](https://huggingface.co/MLRS/BERTu) or [`MLRS/mBERTu`](https://huggingface.co/MLRS/mBERTu).
For details on how pre-training was done see the [`pretrain` directory](pretrain).

The models were trained on Korpus Malti v4.0, which can be accessed through the Hugging Face Hub using [`MLRS/korpus_malti`](https://huggingface.co/datasets/MLRS/korpus_malti).


## Evaluation

These models were evaluated on Dependency Parsing, Part-of-Speech Tagging, Named-Entity Recognition, & Sentiment Analysis.
They can be used to make predictions as follows (using [`finetune` directory](finetune) as the working path):

```shell
allennlp predict $MODEL_PATH $DATA_PATH \
  --predictor $predictor \
  --multitask-head $head \
  --use-dataset-reader \
  --output-file $PREDICTIONS_PATH \
  --include-package modules
```
where `$predictor` & `$head` is specific to the task:

|                          | `$predictor`              | `$head`      |
|--------------------------|---------------------------|--------------|
| Dependency Parsing*      | `depdendency_parser`      | `ud`         |
| Part-of-Speech Tagging   | `part_of_speech_tagger`   | `pos`        |
| Named-Entity Recognition | `named_entity_recogniser` | `ner`        |
| Sentiment Analysis       | `sentiment_classifier`    | `sentiment`  |

_*Additionally also add the following arguments as well: `--extend-namespace head_tags --extend-namespace head_indices`._

For details on how fine-tuning was done see the [`finetune` directory](finetune).
