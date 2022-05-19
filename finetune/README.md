# Fine-tuning

Fine-tuning to a downstream tasks requires a [pre-trained model](../pretrain).
Even if using an off-the-shelf model without any pre-training adjustments, the model should be stored to disk & referenced that way.


## Datasets

The following are the datasets used, & how to access them:

- **[MUDT (UniversalDependencies)](https://github.com/UniversalDependencies/UD_Maltese-MUDT/)**
- **[MLRS POS Gold data](https://mlrs.research.um.edu.mt/)**
  combine the data from the different domains & split them into training, development, & testing files,
  using [the convertor script](../data/mlrs/convert_data.py):
  ```shell
  python mlrs/convert_data.py part_of_speech \
  --path=$LABELLED_DATA_PATH \
  --generated_path="$LABELLED_DATA_PATH/mlrs_pos.tsv" \
  --dev_proportion=0.1 \
  --test_proportion=0.1
  ```
- **[WikiAnn](https://github.com/afshinrahimi/mmner)**
- **[Sentiment Analysis](https://github.com/jerbarnes/typology_of_crosslingual/tree/master/data/sentiment/mt)**


## Training

1. Set the `DATA_PATH` environment variable to the base paths where the labelled datasets are stored.
2. Ensure that the relative paths to the [models](config/lib/models.libsonnet) & [data](config/lib/data.libsonnet) are as specified in the configuration.
3. Run a fine-tuning run by executing the following command:
   ```shell
   finetune.sh ${name}
   ```
   where `${name}` is a [fine-tuning configuration](config/autogen), typically of the format `${task}_${model}`. 
   This will run a number of training & evaluation runs, which are stored in the indicated `output_path` defined in [the script](finetune.sh).


## Acknowledgements

Most of the code here has been adapted from [previous work](https://github.com/ethch18/specializing-multilingual).
