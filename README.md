# BERTu: A BERT-based language model for the Maltese language :malta:

<img src="logo.png" width="200" margin-right="1em" align="left" />

This repository contains code & information relevant for the paper [Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and BERT Models for Maltese](https://aclanthology.org/2022.deeplo-1.10/).

The pre-trained language models can be accessed through the Hugging Face Hub using [`MLRS/BERTu`](https://huggingface.co/MLRS/BERTu) or [`MLRS/mBERTu`](https://huggingface.co/MLRS/mBERTu).
For details on how pre-training was done see the [`pretrain` directory](pretrain).

The models were trained on Korpus Malti v4.0, which can be accessed through the Hugging Face Hub using [`MLRS/korpus_malti`](https://huggingface.co/datasets/MLRS/korpus_malti).

- For details on how fine-tuning was done see the [`finetune` directory](finetune).
- To consume fine-tuned models for evaluation/prediction refer to the [`evaluate` directory](evaluate).

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
