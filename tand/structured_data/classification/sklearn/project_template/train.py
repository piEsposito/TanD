import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from tand.util import *
from model import parse_model_option

import os
import json
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

    # for preliminar testing, will remove later:
    # df = df.drop(df[df[config["train"]["labels_column"]] == 2].index, axis=0)
    # df = df.drop(['Unnamed: 0'], axis=1)

    # label_nbr = len(df[config["train"]["labels_column"]].unique())
    label_names = config["train"]["labels"]

    y = np.array(df[config["train"]["labels_column"]])
    df = df.drop([config["train"]["labels_column"]], axis=1)
    X = np.array(df)

    print(X.shape, y.shape)
    print(df.columns)

    classifier = parse_model_option(config["train"]["model"],
                                    config["train"]["model_args"])

    print(classifier)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.25,
                                                        random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_true = y_test

    metrics = eval_model_per_class(y_true, y_pred, label_names)
    metrics["accuracy"] = (y_pred == y_true).astype(np.float32).mean()
    mlflow.log_params(metrics)

    mlflow.sklearn.log_model(
        sk_model=classifier,
        artifact_path="model",
        registered_model_name=config["mlflow"]["model_name"]
    )

    conf_matrix_fname = save_confusion_matrix(y_true,
                                              y_pred,
                                              label_names)
    mlflow.log_artifact(conf_matrix_fname)
    os.remove(conf_matrix_fname)

    api_request_model = get_request_features(df)
    with open("request_model.json", "w") as rmodel:
        json.dump(api_request_model, rmodel, indent=4)

    # checking if there are any productions models,
    # so we can put at least one in production

    model_name = config['mlflow']['model_name']

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    except:
        client = MlflowClient()
        version = client.search_model_versions(f"name='{model_name}'")[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

    print(metrics)


if __name__ == '__main__':
    main()
