local template = import "template.libsonnet";

local archive_root = std.extVar("MTL_ALLENNLP_OUTPUTS");
local fold = std.extVar("SEED_SET");
local data_root = std.extVar("MTL_DATA");

{
    build_dp(language_model_name)::
    template{
        "key": "ud",
        "model_name": language_model_name,
        "dataset": "universal_dependencies",
        "head": {
            "type": "biaffine_parser",
            "arc_representation_dim": 100,
            "tag_representation_dim": 100,
            // NEW!
            "use_mst_decoding_for_validation": true,
            "dropout": 0.3,
            "input_dropout": 0.3,
            "encoder": {
                "type": "pass_through",
                "input_dim": 768,
            },
            "initializer": {
                "regexes": [
                    [".*projection.*weight", {"type": "xavier_uniform"}],
                    [".*projection.*bias", {"type": "zero"}],
                    [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                    [".*tag_bilinear.*bias", {"type": "zero"}],
                    [".*weight_ih.*", {"type": "xavier_uniform"}],
                    [".*weight_hh.*", {"type": "orthogonal"}],
                    [".*bias_ih.*", {"type": "zero"}],
                    [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                ]
            },
        },
        "batch_size": 128,
        "validation_metric": ["+ud_LAS"],
    },
    build_pos(language_model_name, language_specific=true, dataset="mlrs_pos")::
    template{
        "key": "pos",
        "model_name": language_model_name,
        "dataset": dataset,
        "dataset_reader"+: {
            "readers"+: {
                "pos"+: {
                    "language_specific": language_specific,
                },
            },
        },
        "head": {
            "type": "linear_tagger",
            "encoder": {
                "type": "pass_through",
                "input_dim": 768,
            },
            "dropout": 0.3,
            "initializer": {
                "regexes": [
                    [".*projection.*weight", {"type": "xavier_uniform"}],
                    [".*projection.*bias", {"type": "zero"}],
                    [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                    [".*tag_bilinear.*bias", {"type": "zero"}],
                    [".*weight_ih.*", {"type": "xavier_uniform"}],
                    [".*weight_hh.*", {"type": "orthogonal"}],
                    [".*bias_ih.*", {"type": "zero"}],
                    [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                ]
            },
        },
        "batch_size": 128,
        "validation_metric": ["+pos_accuracy"],
    },
    build_ner(language_model_name, dataset="wikiann")::
    template{
        "key": "ner",
        "model_name": language_model_name,
        "dataset": dataset,
        "head": {
            "type": "crf_tagger",
            "encoder": {
                "type": "pass_through",
                "input_dim": 768,
            },
            "include_start_end_transitions": false,
            // following SciBERT
            "dropout": 0.2,
            "calculate_span_f1": true,
            "constrain_crf_decoding": true,
            "label_encoding": "BIO",
        },
        "batch_size": 64,
        "validation_metric": ["+ner_f1-measure-overall"],
    },
    build_multilevel_ner(language_model_name, dataset="mapa_ner")::
    template{
        "key": ["ner", "ner_level2"],
        "model_name": language_model_name,
        "dataset": dataset,
        "dataset_reader"+: {
            "readers"+: {
                "ner"+: {
                    "tag_level": 1,
                },
                "ner_level2"+: {
                    "tag_level": 2,
                },
            },
        },
        "head": {
            "type": "linear_tagger",
            "encoder": {
                "type": "pass_through",
                "input_dim": 768,
            },
            "dropout": 0.3,
            "initializer": {
                "regexes": [
                    [".*projection.*weight", {"type": "xavier_uniform"}],
                    [".*projection.*bias", {"type": "zero"}],
                    [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                    [".*tag_bilinear.*bias", {"type": "zero"}],
                    [".*weight_ih.*", {"type": "xavier_uniform"}],
                    [".*weight_hh.*", {"type": "orthogonal"}],
                    [".*bias_ih.*", {"type": "zero"}],
                    [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                ]
            },
        },
        "batch_size": 64,
        "validation_metric": "+ner_level2_accuracy",
    },
    build_sa(language_model_name)::
    template{
        "key": "sentiment",
        "model_name": language_model_name,
        "dataset": "sentiment_analysis",
        "head": {
            "type": "linear_classifier",
            "encoder": {
                "type": "pass_through",
                "input_dim": 768,
            },
            "dropout": 0.5,
            "initializer": {
                "regexes": [
                    [".*projection.*weight", {"type": "xavier_uniform"}],
                    [".*projection.*bias", {"type": "zero"}],
                    [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                    [".*tag_bilinear.*bias", {"type": "zero"}],
                    [".*weight_ih.*", {"type": "xavier_uniform"}],
                    [".*weight_hh.*", {"type": "orthogonal"}],
                    [".*bias_ih.*", {"type": "zero"}],
                    [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                ]
            },
        },
        "batch_size": 32,
        "validation_metric": ["+sentiment_fscore"],
        "trainer"+: {
            "optimizer"+: {
                "lr": 1e-4,
            },
        },
    },
}
