import argparse
import operator
import os
from collections import Counter

import tokenizers
import transformers
from tokenizers.implementations import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


def train_bert(data, vocabulary_size):
    tokeniser = BertWordPieceTokenizer(lowercase=False)
    tokeniser.train(files=data,
                    vocab_size=vocabulary_size,
                    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    return tokeniser


def select_tokens(base_tokeniser: transformers.PreTrainedTokenizer,
                  specific_tokeniser: tokenizers.implementations.BaseTokenizer,
                  data,
                  count=99):
    unknown_token = base_tokeniser.unk_token

    def count_unknowns(tokenised_input):
        return sum(1 if unknown_token == item else 0 for item in tokenised_input)

    base_vocabulary = set(base_tokeniser.get_vocab().keys())

    unk_tokens = []  # tokens which are known to the specialised tokeniser, but are unknown by the base tokeniser
    all_tokens = []
    with open(data, encoding="utf-8") as file:
        for line in file:
            for token, _ in specific_tokeniser.pre_tokenizer.pre_tokenize_str(line):
                base_tokens = base_tokeniser.tokenize(token)
                specific_tokens = specific_tokeniser.encode(token).tokens
                all_tokens.extend(specific_tokens)
                if count_unknowns(base_tokens) > count_unknowns(specific_tokens):
                    unk_tokens.extend([token for token in specific_tokens if token not in base_tokens])

    new_tokens = filter(lambda token: token not in base_vocabulary, unk_tokens)
    counter = Counter(new_tokens)
    return list(map(operator.itemgetter(0), counter.most_common(count)))

    def sorted_by_frequency(tokens):
        new_tokens = filter(lambda token: token not in base_vocabulary, tokens)
        counter = Counter(new_tokens)
        return list(map(operator.itemgetter(0), counter.most_common()))

    unk_tokens = sorted_by_frequency(unk_tokens)[:count]
    all_tokens = sorted_by_frequency(all_tokens)

    i = 0
    while len(unk_tokens) < count and i < len(all_tokens):
        if all_tokens[i] not in unk_tokens:
            unk_tokens.append(all_tokens[i])
        i += 1

    return unk_tokens


def load(name):
    model = AutoModelForMaskedLM.from_pretrained(name)
    tokeniser = AutoTokenizer.from_pretrained(name, use_fast=False)

    return model, tokeniser


def save(model, tokeniser, path):
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    if hasattr(tokeniser, "save_pretrained"):
        tokeniser.save_pretrained(path)
    else:
        tokeniser.save_model(path)

        # tokeniser configuration isn't saved by `tokenizers`
        # load default configuration & replace any custom configuration
        configuration = AutoTokenizer.from_pretrained(path)
        tokeniser_parameters = tokeniser._parameters
        configuration.do_lower_case = getattr(tokeniser_parameters, "lowercase", None)
        configuration.save_pretrained(path)


MODEL_TYPE_BERT = "bert"
MODEL_TYPES = {
    MODEL_TYPE_BERT,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="The directory path where the loaded model/tokeniser are to stored. "
                             "Note that any conflicting files will be overwritten.")

    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="The model architecture to use. "
                             "This should be a pre-trained model as defined in https://huggingface.co/models/, "
                             f"or a model architecture type from {list(MODEL_TYPES)}. "
                             "For the latter, the configuration (not the model weights) are specified, "
                             "so that it can be pre-trained from scratch.")

    tokeniser_data_argument = \
        parser.add_argument("--tokeniser_data",
                            type=str,
                            nargs="+",
                            help="The paths to the data used to train the tokeniser. "
                                 "This will be used to train a tokeniser from scratch if the name indicates a new model. "
                                 "Otherwise, the pre-trained tokeniser vocabulary will be used as a base, "
                                 "& the top tokens from a newly trained tokeniser will be augmented. "
                                 "If you want to load a pre-trained tokeniser as is do not specify this option, "
                                 "since it might have an unintended effect.")

    parser.add_argument("--vocabulary_size",
                        type=int,
                        default=52000,
                        help="The vocabulary size. "
                             "Used only if a tokeniser is trained.")

    args = parser.parse_args()

    name = args.name
    try:  # pre-trained model
        model, tokeniser = load(name)
        name = model.base_model_prefix
    except OSError:  # new model
        model, tokeniser = AutoConfig.for_model(name), None
        model.vocab_size = args.vocabulary_size

        if not args.tokeniser_data:
            raise argparse.ArgumentError(tokeniser_data_argument,
                                         "When loading a new model, a tokeniser should be trained.")

    if args.tokeniser_data:
        trained_tokeniser = train_bert(args.tokeniser_data[0], args.vocabulary_size)
        if tokeniser:  # pre-trained model
            tokens = select_tokens(tokeniser, trained_tokeniser, args.tokeniser_data[-1])
            for token, _ in sorted(tokeniser.get_vocab().items(), key=operator.itemgetter(1)):
                if len(tokens) == 0:  # no more tokens to add
                    break

                if token not in tokeniser.all_special_tokens:
                    tokeniser.vocab[tokens.pop()] = tokeniser.vocab.pop(token)
        else:  # new model
            tokeniser = trained_tokeniser

    save(model, tokeniser, args.path)


if __name__ == "__main__":
    main()
