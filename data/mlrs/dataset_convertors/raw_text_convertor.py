import json
from typing import List, IO

from jsonpath_ng import parse

from .dataset_convertor import DatasetConvertor, Instance, INSTANCE_REFERENCE


class RawTextConvertor(DatasetConvertor):
    """
    Converts raw text from documents.
    The source is a file of JSON-lines format, where each line is an object of the following form:

    ``{"text": ["A sentence.", "Another sentence", ...]}``

    Sentences are converted to a line-by-line format, where each sentence takes up a line.
    Each sentence is treated as a separate :class:`Instance`.
    """

    @property
    def source_file_regex(self) -> str:
        return "*.jsonl"

    @property
    def target_file_extension(self) -> str:
        return ".txt"

    def parse(self, source_file: IO) -> List[dict]:
        return [json.loads(line) for line in source_file]

    def index(self, data: List[dict]) -> List[INSTANCE_REFERENCE]:
        references = []
        for i, document in enumerate(data):
            sentence_key = "text"
            for j, sentence in enumerate(document[sentence_key]):
                references.append([i, f"{sentence_key}", f"[{j}]"])
        return references

    def convert(self, data: List[dict], reference: INSTANCE_REFERENCE, key: str) -> str:
        document_index, sentence_reference = reference[0], reference[1:]
        return parse(".".join(sentence_reference)).find(data[document_index])[0].value.strip().replace("\n", " ") + "\n"
