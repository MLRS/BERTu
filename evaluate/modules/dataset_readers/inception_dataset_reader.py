import logging
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("inception")
class InceptionDatasetReader(DatasetReader):
    """
    Reads instances which are in INCEpTION format.
    Every instance is separated by a blank line, with a single instance spanning multiple lines.
    Every instance may have an optional comment which start with a "#" character.

    The remainder of the lines of instance are TSV values.
    It is assumed that each line contains the token index & the token as its first 2 values,
    whilst the rest of the values are the tags.
    By default, the first tag is taken as the label, but can be overridden by `tag_index`.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels",
                 tag_index: int = 0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        self.tag_index = tag_index

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, encoding="utf-8") as data_file:
            logger.info(
                f"Reading instances from lines in file at: {file_path}"
            )

            tokens = []
            labels = []
            for line in data_file:
                line = line.strip()
                if not line:  # end of instance delimiter
                    yield self.text_to_instance(tokens, labels)
                    tokens = []
                    labels = []
                elif line.startswith("#"):  # comments
                    continue
                else:
                    token_index, token, tags = line.split("\t", maxsplit=2)
                    tags = tags.split("\t")
                    tokens.append(Token(token))
                    labels.append(tags[self.tag_index])

            if tokens:  # last instance in file without proceeding instance delimiter
                yield self.text_to_instance(tokens, labels)

    @overrides
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        token_field = TextField(tokens)
        tag_field = SequenceLabelField(tags, token_field, self.label_namespace)
        meta_field = MetadataField({"words": [token.text for token in tokens]})

        fields = {
            "tokens": token_field,
            "tags": tag_field,
            "metadata": meta_field,
        }
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers
