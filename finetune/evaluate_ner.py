import argparse
import importlib
import pkgutil
from typing import Dict

import torch
from allennlp.data.fields import MetadataField
from allennlp.models import load_archive
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure
from tqdm import tqdm

import finetune.modules


def import_submodules(module):
    for loader, module_name, is_pkg in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
        importlib.import_module(module_name)


import_submodules(finetune.modules)


def encode(tags, vocabulary, namespace):
    return torch.Tensor([vocabulary.get_token_index(tag, namespace) for tag in tags])


def one_hot_encode(tags, vocabulary, namespace):
    return torch.Tensor(
        [[1 if index == label else 0 for index in range(vocabulary.get_vocab_size(namespace))] for label in
         encode(tags, vocabulary, namespace)])


def evaluate(data_path: str, model_path: str) -> Dict[str, Dict[str, float]]:
    archive = load_archive(model_path)
    model = archive.model
    dataset_reader = archive.dataset_reader
    vocabulary = model.vocab

    results = {}
    for head in model._heads.keys():
        span_f1 = SpanBasedF1Measure(vocabulary, tag_namespace=head)

        for gold_instance in tqdm(dataset_reader.readers[head].read(data_path), desc=f"Evaluating {head}"):
            gold_instance.fields["task"] = MetadataField(head)
            prediction = model.forward_on_instance(gold_instance)

            span_f1(one_hot_encode(prediction[f"{head}_tags"], vocabulary, head).unsqueeze(0),
                    encode(gold_instance["tags"], vocabulary, head).unsqueeze(0))

        results[head] = span_f1.get_metric()
        print(f"{head} results:", results[head])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a multi-task model on Named-Entity Recognition."
    )

    parser.add_argument("model_path",
                        type=str,
                        help="The path to the model to evaluate.")

    parser.add_argument("data_path",
                        type=str,
                        help="The path where the evaluation data is located.")

    args = parser.parse_args()

    print(evaluate(args.data_path, args.model_path))


if __name__ == '__main__':
    main()
