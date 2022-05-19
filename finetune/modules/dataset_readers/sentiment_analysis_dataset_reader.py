import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    TextField,
    LabelField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("sentiment_analysis")
class SentimentAnalysisDatasetReader(DatasetReader):
    """
    Reads a file in CSV format, with 2 expected columns: the label followed by the text.
    Although the text may itself contain "," characters, these will be considered as part of the text.
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            logger.info(
                "Reading sentiment instances from dataset at: %s", file_path
            )

            for line in file:
                label, text = line.split(",", maxsplit=1)
                label = int(label)
                yield self.text_to_instance(text, label)

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: int) -> Instance:
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = [Token(text)]

        fields["words"] = TextField(tokens)
        fields["sentiment"] = LabelField(
            label, label_namespace="sentiment", skip_indexing=True
        )

        fields["metadata"] = MetadataField({"words": text, "label": label})
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["words"].token_indexers = self._token_indexers
