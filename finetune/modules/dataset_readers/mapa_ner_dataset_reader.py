import logging

from allennlp.data.dataset_readers import DatasetReader

from .inception_dataset_reader import InceptionDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("mapa_ner")
class MapaNERDatasetReader(InceptionDatasetReader):
    """
    Reads instances from the MAPA NER data.
    The tags are the Level 1 & Level 2 BIO tags (according to the MAPA hierarchy).
    """

    def __init__(self,
                 tag_level: int = 1,
                 **kwargs) -> None:
        assert tag_level in (1, 2)
        super().__init__(tag_index=tag_level - 1, **kwargs)
