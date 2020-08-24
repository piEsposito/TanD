import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from tand.util import *
from lib import preprocess, Net

import os
import json
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

with open("config.json", "r") as file:
    config = json.load(file)

# mlflow.set_tracking_uri("sqlite:///database.db")

# checking if experiment exists, then creating with proper artifact logging path:
exp_ = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])

if exp_ is None:
    mlflow.create_experiment(name=config["mlflow"]["experiment_name"],
                             artifact_location=os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"])

mlflow.set_experiment(experiment_name=config["mlflow"]["experiment_name"])

exp = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
print(exp)


def main():
    df = pd.read_csv(config["train"]["data_path"])
    y = np.array(df[config["train"]["label_column"]])

    df = preprocess(df)

    label_nbr = len(df[config["train"]["label_column"]].unique())
    label_names = config["train"]["label"]

    y = np.array(df[config["train"]["label_column"]])
    df = df.drop([config["train"]["label_column"]] + config['train']['to_drop'], axis=1)
    X = np.array(df)

    print(X.shape, y.shape)

    try:
        device = torch.device(config["train"]["device"])
    except:
        device = torch.device("cpu")

    classifier = Net(input_dim=df.shape[1],
                     hidden_dim=config["train"]["hidden_dim"]).to(device)
    criterion = torch.nn.functional.mse_loss

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.25,
                                                        random_state=42)

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    # create dataloader with specified batch_size
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train,
                                                   batch_size=config["train"]["batch_size"],
                                                   shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(ds_test,
                                                  batch_size=config["train"]["batch_size"],
                                                  shuffle=True)

    trainer = Trainer(classifier, device=device, criterion=criterion)
    trainer.train(dataloader_train,
                  dataloader_test,
                  config["train"]["epochs"],
                  config["train"]["log_every"],
                  task="regression")

    # eval step

    metrics = {}
    metrics["mse"] = trainer.metric
    mlflow.log_params(metrics)

    mlflow.pytorch.log_model(
        pytorch_model=classifier,
        artifact_path="model",
        registered_model_name=config["mlflow"]["model_name"]
    )

    api_request_model = get_request_features(df)
    with open("request_model.json", "w") as rmodel:
        json.dump(api_request_model, rmodel, indent=4)

    # checking if there are any productions models,
    # so we can put at least one in production

    model_name = config['mlflow']['model_name']

    try:
        mlflow.pytorch.load_model(f"models:/{model_name}/Production")
    except:
        client = MlflowClient()
        version = client.search_model_versions(f"name='{model_name}'")[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )


if __name__ == '__main__':
    main()
