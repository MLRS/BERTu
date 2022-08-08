# BERTu: A BERT-based language model for the Maltese language :malta:

This repository contains code & information relevant for the paper [Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and BERT Models for Maltese](https://aclanthology.org/2022.deeplo-1.10/).

The pre-trained language models can be accessed through the Hugging Face Hub using [`MLRS/BERTu`](https://huggingface.co/MLRS/BERTu) or [`MLRS/mBERTu`](https://huggingface.co/MLRS/mBERTu).
For details on how pre-training was done see the [`pretrain` directory](pretrain).

The models were trained on Korpus Malti v4.0, which can be accessed through the Hugging Face Hub using [`MLRS/korpus_malti`](https://huggingface.co/datasets/MLRS/korpus_malti).


## Evaluation

These models were evaluated on Dependency Parsing, Part-of-Speech Tagging, Named-Entity Recognition, & Sentiment Analysis.
To make predictions, use [`finetune` directory](finetune) as the working path (installing the latest [AllenNLP](https://github.com/allenai/allennlp) instead of the one specified), & execute the following command:

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

|                          | `$name`                                                                                                   | `$predictor`              | `$head`      |
|--------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------|--------------|
| Dependency Parsing       | [BERTu-ud](https://huggingface.co/MLRS/BERTu-ud)                                                          |`depdendency_parser`      | `ud`         |
| Part-of-Speech Tagging   | [BERTu-xpos](https://huggingface.co/MLRS/BERTu-xpos)/[BERTu-upos](https://huggingface.co/MLRS/BERTu-upos) |`part_of_speech_tagger`   | `pos`        |
| Named-Entity Recognition | [BERTu-ner](https://huggingface.co/MLRS/BERTu-ner)                                                        |`named_entity_recogniser` | `ner`        |
| Sentiment Analysis       | [BERTu-sentiment](https://huggingface.co/MLRS/BERTu-sentiment)                                            |`sentiment_classifier`    | `sentiment`  |

For details on how fine-tuning was done see the [`finetune` directory](finetune).

## Citation

Cite this work as follows: 

```bibtex
@inproceedings{BERTu,
    title = "Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and {BERT} Models for {M}altese",
    author = "Micallef, Kurt  and
              Gatt, Albert  and
              Tanti, Marc  and
              van der Plas, Lonneke  and
              Borg, Claudia",
    booktitle = "Proceedings of the Third Workshop on Deep Learning for Low-Resource Natural Language Processing",
    month = jul,
    year = "2022",
    address = "Hybrid",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.deeplo-1.10",
    doi = "10.18653/v1/2022.deeplo-1.10",
    pages = "90--101",
}
```
