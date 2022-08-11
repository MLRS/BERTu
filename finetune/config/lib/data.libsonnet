local base_path = std.extVar("DATA_PATH");

{
    "universal_dependencies": {
        "path": base_path + "/MUDT/mt_mudt-ud-%s.conllu",
        "size": 1123,
    },
    "mlrs_pos": {
        "path": base_path + "/MLRS POS Gold/mlrs_pos-%s.tsv",
        "size": 4935,
    },
    "mapa_ner": {
        "path": base_path + "/MAPA NER/mapa_ner-%s.tsv",
        "size": 267896,
    },
    "wikiann": {
        "path": base_path + "/WikiAnn/mt/%s",
        "size": 100,
    },
    "sentiment_analysis": {
        "path": base_path + "/Maltese Sentiment/%s.csv",
        "size": 595,
    },
}
