from typing import Dict

from .dataset_convertor import DatasetConvertor, Instance
from .entity_recognition_convertor import EntityRecognitionConvertor
from .part_of_speech_convertor import PartOfSpeechConvertor
from .raw_text_convertor import RawTextConvertor

CONVERTORS: Dict[str, DatasetConvertor] = {
    "entity_recognition": EntityRecognitionConvertor(),
    "part_of_speech": PartOfSpeechConvertor(),
    "raw_text": RawTextConvertor(),
}