from typing import Dict

from .dataset_convertor import DatasetConvertor, Instance
from .part_of_speech_convertor import PartOfSpeechConvertor
from .raw_text_convertor import RawTextConvertor

CONVERTORS: Dict[str, DatasetConvertor] = {
    "part_of_speech": PartOfSpeechConvertor(),
    "raw_text": RawTextConvertor(),
}