import mlflow
import mlflow.sklearn

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np

import json
import os

app = FastAPI()

with open("config.json", "r") as file:
    config = json.load(file)

with open("request_model.json", "r") as file:
    request_model = json.load(file)

model_name = config['mlflow']['model_name']
model_stage = os.environ["MODEL_STAGE"]
label_names = config["train"]["labels"]
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")

API_TOKEN = os.environ["API_TOKEN"]


async def authenticate(request: Request):
    global API_TOKEN
    token = request.headers.get("TOKEN")
    if (token is None) or (token != API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {}


async def validate_json(request: Request):
    # makes the requests keys to keep the same order as the model

    global request_model
    features = {}

    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Bad request")

    for key in body.keys():
        if request_model.get(key) is None:
            raise HTTPException(status_code=400, detail="Bad request")
        if type(body[key]) not in [float, int]:
            raise HTTPException(status_code=400, detail="Bad request")

    for key in request_model.keys():
        if body.get(key) is None:
            raise HTTPException(status_code=400, detail="Bad request")
        features[key] = body[key]

    return features


@app.post("/predict")
async def root(token: dict = Depends(authenticate), features: dict = Depends(validate_json)):
    global model, label_names, device
    as_df = pd.DataFrame([features])
    arr = np.array(as_df)
    pred = model.predict(arr).item()

    pred = label_names[int(pred)]
    return JSONResponse(
        status_code=200,
        content={"prediction": pred}
    )


@app.post("/update-model")
async def update_model(token: dict = Depends(authenticate)):
    global model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")
    return JSONResponse(
        status_code=200,
        content={"message": "model updated"}
    )


@app.get("/")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"message": "alive and running!"}
    )


@app.exception_handler(Exception)
async def handle_exception(*args):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )
