# TanD PyTorch Structured data classification template
This file documents how the project template for structured data classification works, what can be changes and how.

## `config.json`
The `config.json` file has the following structure:
```json
{
  "train": {
    "__help": "Configurations for the training of the project (of the model, etc...)",
    "data_path": "data/data.csv",
    "labels_column": "target",

    "log_every": 250,
    "epochs": 50,

    "hidden_dim": 256,
    "batch_size": 32,
    "device": "cpu",
    "labels": ["no_heart_disease", "heart_disease"],
    "to_drop": []

  },


  "mlflow": {
    "__help": "Configurations for the mlflow model manager of the project",
    "model_name": "pytorch-classifier-nn-heart-disease",
    "experiment_name": "heart-disease-experiment"
  }
}
```
The variables set here are explained below:

`train`:
 * `data_path`: path for the csv file with data - expected to have continuous or one-hot-encoded data.
 * `labels_column`: column on the csv file which contains the labels.
 * `labels`: names for each of the labels, to be used by the API

 * `to_drop`: columns to be dropped of the DF
 * `device`: device where the training will occur.
 * `hidden_dim, batch_size, epochs, log_every` - autoexplanatory parameters for training.
 
 `mlflow`:
  * `model_name` : model name to be set in `mlflow`
  * `experiment_name`: experiment name to be set in `mlflow`
 
 
## `train.py`
The `train.py` file is very important on a `tand` project. It defines the training process for the data and, although it works with no-coding or changes, is highly flexible, as is does not use strange data structures and very specific stuff. 

We will enumerate the role of some files on that process:

* `lib/model.py`: defines the `torch` model that will be used for training, parametrized by `config.json`. You can change it anyway you want but, to comply with `tand.util.Trainer` it should have the proper number of output nodes. If you change it there, just put some `*args, **kwargs` not to worry with undeclared arguments.

* `lib/preprocess.py`: before training, the df is turned into the result of this function. It is intended for preprocessing needed that uses more stuff that only dropping columns, as it can be set on `config.json`. 

After preprocessing the df, it saves a `json` file with the structure of the dataframe as labels. This is used both for testing the model and for validating requests bodies and reordering them to comply with the columns order as it was on train time.

It stores the model and some evaluation metrics at `mlflow`, so you can analyze what you want to put into production.

## Environment variables

`tand` uses environment variables to set `mlflow` related pathes and security at deploy time:

 * MLFLOW_TRACKING_URI, defaults to "sqlite:///database.db". URL to the DB which will backend `mlflow`. Can be set to remote dbs.
 * MLFLOW_DEFAULT_ARTIFACT_ROOT, defaults to "./mlruns/". Path where the model will be stored at `mlflow`. If AWS credentials are set properly, supports `s3` urls. 
 * MODEL_STAGE, default to "Production". Stage of the model which `mlflow` will fetch at production time.
 * API_TOKEN, token which the API will use for authentication
 
 For information of the AWS related variables and config files, check documentation for `tand.deployment`.
 