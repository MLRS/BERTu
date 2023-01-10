import logging
from typing import List

from allennlp.data import Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from overrides import overrides

from .inception_dataset_reader import InceptionDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("mlrs_pos")
class MlrsPosDatasetReader(InceptionDatasetReader):
    """
    Reads instances from the MLRS POS data.
    The tags are the language-universal (UPOS) & language-specific (XPOS) tags.
    """

    def __init__(self,
                 language_specific: bool = True,
                 **kwargs) -> None:
        super().__init__(tag_index=1 if language_specific else 0, **kwargs)

    @overrides
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        token_field = TextField(tokens)
        tag_field = SequenceLabelField(tags, token_field, label_namespace="pos")
        meta_field = MetadataField({"words": [token.text for token in tokens]})

        fields = {
            "tokens": token_field,
            "pos_tags": tag_field,
            "metadata": meta_field,
        }
        return Instance(fields)
