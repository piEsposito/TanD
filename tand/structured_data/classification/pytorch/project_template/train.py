import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from tand.util import *
from model import Net
from preprocess import preprocess

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
    y = np.array(df[config["train"]["labels_column"]])

    df = preprocess(df)

    label_nbr = len(df[config["train"]["labels_column"]].unique())
    label_names = config["train"]["labels"]

    y = np.array(df[config["train"]["labels_column"]])
    df = df.drop([config["train"]["labels_column"]], axis=1)
    X = np.array(df)

    print(X.shape, y.shape)
    # df = df.drop(['Unnamed: 0'], axis=1)

    try:
        device = torch.device(config["train"]["device"])
    except:
        device = torch.device("cpu")

    classifier = Net(input_dim=df.shape[1],
                     output_dim=label_nbr,
                     hidden_dim=config["train"]["hidden_dim"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()

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
                  config["train"]["log_every"])

    # eval step

    y_true, y_pred, scores = get_preds_labels_scores(dataloader_test, classifier, device)

    metrics = eval_model_per_class(y_true, y_pred, label_names)
    metrics["accuracy"] = trainer.acc
    mlflow.log_params(metrics)

    mlflow.pytorch.log_model(
        pytorch_model=classifier,
        artifact_path="model",
        registered_model_name=config["mlflow"]["model_name"]
    )

    conf_matrix_fname = save_confusion_matrix(y_true,
                                              y_pred,
                                              label_names)
    mlflow.log_artifact(conf_matrix_fname)
    os.remove(conf_matrix_fname)

    roc_curve_fname = save_roc_curve(y_true,
                                     scores,
                                     label_names)
    mlflow.log_artifact(roc_curve_fname)
    os.remove(roc_curve_fname)

    pr_curve_fname = save_pr_curve(y_true,
                                   scores,
                                   label_names)
    mlflow.log_artifact(pr_curve_fname)
    os.remove(pr_curve_fname)

    api_request_model = get_request_features(df)
    with open("request_model.json", "w") as rmodel:
        json.dump(api_request_model, rmodel, indent=4)

    # checking if there are any productions models,
    # so we can put at least one in production

    model_name = config['mlflow']['model_name']

    try:
        model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
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
