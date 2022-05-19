from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, List, IO

DATA = TypeVar("DATA")

INSTANCE_REFERENCE = List[Any]


@dataclass
class Instance:
    """
    An instance in a dataset, identified by the source file and some reference specific to the dataset.
    """
    file_path: str
    reference: INSTANCE_REFERENCE


class DatasetConvertor(ABC, Generic[DATA]):
    """
    A dataset convertor for a given dataset, which:

    1. Reads files into ``DATA`` structures.
    2. Converts an individual :class:`Instance` into a standard format.
    """

    @property
    def source_file_regex(self) -> str:
        """
        The file regex indicating the input file names to match.

        Should be overridden to filter out any additional files which are not part of the dataset.
        """
        return "*"

    @property
    def target_file_extension(self) -> str:
        """
        The output file extension to use.
        """
        return ""

    def parse(self, source_file: IO) -> Generic[DATA]:
        """
        Parses the given source file.

        :param source_file: The file buffer to parse.
        :return: The parsed data structure.
        """
        return source_file

    @abstractmethod
    def index(self, data: Generic[DATA]) -> List[INSTANCE_REFERENCE]:
        """
        Indexes the given portion of data.

        :param data: The input data representation.
        :return: The reference for each :class:`Instance` in the `data`.
        """
        ...

    @abstractmethod
    def convert(self, data: Generic[DATA], reference: INSTANCE_REFERENCE, key: str) -> str:
        """
        :param data: The input data representation.
        :param reference: Identified the :class:`Instance` to be converted, relative to the `data`.
        :param key: Provides additional meta-data information useful to trace-back information to the source.
        :return: The converted :class:`Instance` line(s) as they should be persisted in the file,
                 including any line breaks.
        """
        ...
