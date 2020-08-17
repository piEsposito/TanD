eval "$(conda shell.bash hook)"
conda activate blitz;

mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT
