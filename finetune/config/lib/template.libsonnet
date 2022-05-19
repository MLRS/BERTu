local batch_sizes = import "batch_sizes.libsonnet";
local common = import "common.libsonnet";
local models = import "models.libsonnet";
local data = import "data.libsonnet";

local required(name) = '"%s" must be overriden!' % name;

local CUDA_VISIBLE_DEVICES = std.extVar("CUDA_VISIBLE_DEVICES");
local cuda_devices = if std.length(CUDA_VISIBLE_DEVICES) == 0 then [-1]
                     else [std.parseInt(device) for device in std.split(CUDA_VISIBLE_DEVICES, ",")];
local max_length = 512;

{
    local template = self,

    key:: error required("key"),
    dataset:: error required("dataset"),
    model_name:: error required("model_name"),
    head:: error required("head"),
    batch_size:: error required("batch_size"),
    validation_metric:: ["+.sum"],

    local model_path = models[template.model_name],
    local device_batch_size = template.batch_size / std.length(cuda_devices),
    local data_size = data[template.dataset]["size"],

    "numpy_seed": common["numpy_seed"],
    "pytorch_seed": common["pytorch_seed"],
    "random_seed": common["random_seed"],
    "dataset_reader": {
        "type": "multitask",
        "readers": {
            [template.key]: {
                "type": template.dataset,
                "token_indexers": {
                    "transformer": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": model_path,
                        "max_length": max_length,
                    },
                },
            },
        },
    },
    "train_data_path": {
        [template.key]: data[template.dataset]["path"] % "train"
    },
    "validation_data_path": {
        [template.key]: data[template.dataset]["path"] % "dev",
    },
    "model": {
        "type": "multitask",
        "arg_name_mapping": {
            "backbone": {
                "tokens": "text",
                "words": "text",
            },
        },
        "backbone": {
            "type": "embedder_and_mask",
            "text_field_embedder": {
                "token_embedders": {
                    "transformer": {
                        "type": "pretrained_transformer_mismatched_with_dropout",
                        "model_name": model_path,
                        "max_length": max_length,
                        "last_layer_only": false,
                        "train_parameters": true,
                        "layer_dropout": 0.1,
                    },
                },
            },
        },
        "heads": {
            [template.key]: template.head,
        },
    },
    "data_loader": {
        "type": "multitask",
        [if template.dataset == "mapa_ner" then "max_instances_in_memory" else null]: {
            "ner": device_batch_size,
        },
        "shuffle": true,
        "scheduler": {
            "type": "unbalanced_homogeneous_roundrobin",
            "batch_size": device_batch_size,
            "dataset_sizes": {
                [template.key]: device_batch_size,
            },
        },
    },
    "validation_data_loader": {
        "type": "multitask",
        [if template.dataset == "mapa_ner" then "max_instances_in_memory" else null]: {
            "ner": device_batch_size,
        },
        "shuffle": true,
        "scheduler": {
            "type": "homogeneous_roundrobin",
            "batch_size": device_batch_size,
        },
    },
    "trainer": {
        "num_epochs": 200,
        "grad_norm": 5.0,
        "patience": 20,
        "cuda_device": cuda_devices[0],
        "validation_metric": template.validation_metric,
        "callbacks": [
            {
                "type": "tensorboard",
                "tensorboard_writer": {
                    "should_log_learning_rate": true,
                    "should_log_parameter_statistics": true,
                },
            },
        ],
        "optimizer": {
            "type": "huggingface_adamw",
            // faster LR; slower one computed via decay_factor in the
            // scheduler
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            // HF says that BERT doesn't use this
            "correct_bias": false,
            // eps kept default
            "parameter_groups": [
                [
                    [
                        "text_field_embedder.*transformer_model.embeddings.*_embeddings.*",
                        "text_field_embedder.*transformer_model.encoder.*.(key|query|value|dense).weight",
                    ],
                    {}
                ],
                [
                    // adapted from SciBERT
                    [
                        "text_field_embedder.*transformer_model.embeddings.LayerNorm.*",
                        "text_field_embedder.*transformer_model.encoder.*.output.LayerNorm.*",
                        "text_field_embedder.*transformer_model.encoder.*.(key|query|value|dense).bias",
                        "text_field_embedder.*transformer_model.pooler.dense.bias",
                    ],
                    {"weight_decay": 0.0}
                ],
                [
                    [
                        "text_field_embedder.*._scalar_mix.*",
                        "text_field_embedder.*transformer_model.pooler.dense.weight",
                        "_head_sentinel",
                        "head_arc_feedforward._linear_layers.*.weight",
                        "child_arc_feedforward._linear_layers.*.weight",
                        "head_tag_feedforward._linear_layers.*.weight",
                        "child_tag_feedforward._linear_layers.*.weight",
                        "arc_attention._weight_matrix",
                        "tag_bilinear.weight",
                        "tag_projection_layer._module.weight",
                        "crf",
                        "linear.weight",
                        "tagger_linear.weight",
                    ],
                    {}
                ],
                [
                    [
                        "head_arc_feedforward._linear_layers.*.bias",
                        "child_arc_feedforward._linear_layers.*.bias",
                        "head_tag_feedforward._linear_layers.*.bias",
                        "child_tag_feedforward._linear_layers.*.bias",
                        "arc_attention._bias",
                        "tag_bilinear.bias",
                        "tag_projection_layer._module.bias",
                        "linear.bias",
                        "tagger_linear.bias",
                    ],
                    {"weight_decay": 0.0},
                ],
            ],
        },
        "learning_rate_scheduler": {
            "type": "ulmfit_sqrt",
            "model_size": 1, // UDify did this so...?
            "affected_group_count": 2,
            // language-specific one epoch
            "warmup_steps": std.ceil(data_size / template.batch_size),
            // language-specific one epoch, by suggestion of UDify
            // https://github.com/Hyperparticle/udify/issues/6
            "start_step": std.ceil(data_size / template.batch_size),
            "factor": 5.0, // following UDify
            "gradual_unfreezing": true,
            "discriminative_fine_tuning": true,
            "decay_factor": 0.05, // yields a slow LR of 5e-5
            // steepness kept to 0.5 (sqrt)
        },
    },
    [if std.length(cuda_devices) > 1 then "distributed" else null]: {
        "cuda_devices": cuda_devices,
    },
}
