from typing import List, IO

from .dataset_convertor import DatasetConvertor, INSTANCE_REFERENCE


class PartOfSpeechConvertor(DatasetConvertor):
    """
    Converts Part-of-Speech tagged sentences.
    Each source line contains the token & the XPOS annotation seperated by a "\t" character.

    This convertor also produces the UPOS tags from the XPOS tags, using the mapping from MUDT.
    The target format is INCEpTION format where each line is of the in the following form:

    ``{token index}\t{token}\t{UPOS tag}\t{XPOS tag}``
    """

    _sentence_separators = ["<s>", "</s>", "<doc>", "</doc>"]

    _upos_mapping = {
        "ADJ": "ADJ",
        "ADV": "ADV",
        "COMP": "SCONJ",
        "CONJ_CORD": "CCONJ",
        "CONJ_SUB": "SCONJ",
        "DEF": "DET",
        "FOC": "ADV",
        "FUT": "AUX",
        "GEN": "ADP",
        "GEN_DEF": "ADP",
        "GEN_PRON": "PRON",
        "HEMM": "VERB",
        "INT": "INTJ",
        "KIEN": "AUX",
        "LIL": "ADP",
        "LIL_PRON": "PRON",
        "LIL_DEF": "ADP",
        "NEG": "PART",
        "NOUN": "NOUN",
        "NOUN_PROP": "PROPN",
        "NUM_CRD": "NUM",
        "NUM_FRC": "NUM",
        "NUM_ORD": "NUM",
        "NUM_WHD": "NUM",
        "PART_ACT": "VERB",
        "PART_PASS": "VERB",
        "PREP": "ADP",
        "PREP_DEF": "ADP",
        "PREP_PRON": "PRON",
        "PROG": "AUX",
        "PRON_DEM": "PRON",
        "PRON_DEM_DEF": "PRON",
        "PRON_INDEF": "PRON",
        "PRON_INT": "PRON",
        "PRON_PERS": "PRON",
        "PRON_PERS_NEG": "AUX",
        "PRON_REC": "PRON",
        "PRON_REF": "PRON",
        "QUAN": "DET",
        "VERB": "VERB",
        "VERB_PSEU": "VERB",
        "X_ABV": "NOUN",
        "X_BOR": "SYM",
        "X_DIG": "NUM",
        "X_ENG": "X",
        "X_FOR": "X",
        "X_PUN": "PUNCT",
    }

    @property
    def source_file_regex(self) -> str:
        return "*.vrt"

    @property
    def target_file_extension(self) -> str:
        return ".tsv"

    def parse(self, source_file: IO) -> List[str]:
        return source_file.readlines()

    def index(self, data: List[str]) -> List[INSTANCE_REFERENCE]:
        references = []
        start_line_index = 0
        is_instance_empty = True
        for i, line in enumerate(data):
            line = line.strip()
            is_instance_empty &= len(line) == 0
            if any(separator in line for separator in self._sentence_separators):
                if start_line_index < i and not is_instance_empty:
                    references.append([start_line_index, i - 1])  # omit current line separator

                # prepare state for next sentence (if any)
                start_line_index = i + 1
                is_instance_empty = True

        if start_line_index < i and not is_instance_empty:
            references.append([start_line_index, i])

        return references

    def convert(self, data: List[str], reference: INSTANCE_REFERENCE, key: str) -> str:
        start_line_index, end_line_index = reference
        return f"# {key}_{start_line_index + 1}\n" \
               + "".join(self.convert_line(line, i)
                         for i, line in enumerate(data[start_line_index:end_line_index + 1])) \
               + "\n"

    def convert_line(self, line: str, line_index: int) -> str:
        parts = line.strip().split("\t")
        assert len(parts) == 2, f"Encountered an error while converting line: \"{line}\""

        word, xpos = parts
        return f"{line_index + 1}\t{word}\t{self._upos_mapping[xpos]}\t{xpos}\n"
