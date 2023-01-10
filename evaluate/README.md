# Evaluation

Evaluating a fine-tuned model requires a [fine-tuned model](../finetune) to make predictions on some data.
You can use either a model which we make available or your custom model.
After installing the [external dependencies](requirements.txt) (which may be different than the [fine-tuning dependencies](../finetune/requirements.txt)), execute the following command (using the current directory as the working directory):

```shell
allennlp predict hf://MLRS/$name $DATA_PATH \
  --predictor $predictor \
  --multitask-head $head \
  --use-dataset-reader \
  --output-file $PREDICTIONS_PATH \
  --include-package modules
```

where `$name`, `$predictor`, & `$head` are specific to the task.
For BERTu these are as follows:

|                          | `$name`                                                                                                   | `$predictor`              | `$head`     |
|--------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------|-------------|
| Dependency Parsing       | [BERTu-ud](https://huggingface.co/MLRS/BERTu-ud)                                                          | `depdendency_parser`      | `ud`        |
| Part-of-Speech Tagging   | [BERTu-xpos](https://huggingface.co/MLRS/BERTu-xpos)/[BERTu-upos](https://huggingface.co/MLRS/BERTu-upos) | `part_of_speech_tagger`   | `pos`       |
| Named-Entity Recognition | [BERTu-ner](https://huggingface.co/MLRS/BERTu-ner)                                                        | `named_entity_recogniser` | `ner`       |
| Sentiment Analysis       | [BERTu-sentiment](https://huggingface.co/MLRS/BERTu-sentiment)                                            | `sentiment_classifier`    | `sentiment` |

Notes:
- The `include-package` references the [code used for making predictions](modules).
  This is mostly a copy of the [fine-tuning code](../finetune/modules), but also includes [predictors](modules/predictors) & certain changes due to [incompatibilities with AllenNLP versions](https://github.com/allenai/allennlp/pull/5676).
- `use-dataset-reader` assumes that the data to be ingested is formatted as the data used to train. 
  So the input data should be formatted as such, including dummy labels.
