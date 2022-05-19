local seed_set = std.extVar("SEED_SET");
local seeds = {
    "allentune": {
        "numpy_seed": std.extVar("NUMPY_SEED"),
        "pytorch_seed": std.extVar("PYTORCH_SEED"),
        "random_seed": std.extVar("RANDOM_SEED"),
    },
    "0": {
        "numpy_seed": 1337,
        "pytorch_seed": 133,
        "random_seed": 13370,
    },
    "1": {
        "numpy_seed": 337,
        "pytorch_seed": 33,
        "random_seed": 3370,
    },
    "2": {
        "numpy_seed": 1537,
        "pytorch_seed": 153,
        "random_seed": 15370,
    },
    "3": {
        "numpy_seed": 2460,
        "pytorch_seed": 246,
        "random_seed": 24601,
    },
    "4": {
        "numpy_seed": 1279,
        "pytorch_seed": 127,
        "random_seed": 12790,
    },
};

seeds[seed_set]
