import csv

import datasets

XPOS_TO_UPOS_MAPPING = {
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

URLS = {
    "train": "train.tsv",
    "validation": "validation.tsv",
    "test": "test.tsv",
}


class MlrsPos(datasets.GeneratorBasedBuilder):
    # VERSION = datasets.Version("2.7.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            # version=VERSION,
            # description=_DESCRIPTION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "xpos": datasets.Sequence(
                        datasets.features.ClassLabel(names=list(XPOS_TO_UPOS_MAPPING.keys()))
                    ),
                    "upos": datasets.Sequence(
                        datasets.features.ClassLabel(names=list(set(XPOS_TO_UPOS_MAPPING.values())))
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://universaldependencies.org/",
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        id = 0
        for path in filepath:
            with open(path, "r", encoding="utf-8") as file:
                data = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                tokens = []
                xpos_tags = []
                upos_tags = []
                for row in data:
                    if len(row) >= 2:
                        token, xpos = row
                        tokens.append(token)
                        xpos_tags.append(xpos)
                        upos_tags.append(XPOS_TO_UPOS_MAPPING[xpos])
                    if len(row) == 0 and len(tokens) > 0:
                        yield id, {
                            "tokens": tokens,
                            "xpos": xpos_tags,
                            "upos": upos_tags,
                        }
                        id += 1
                        tokens = []
                        xpos_tags = []
                        upos_tags = []
