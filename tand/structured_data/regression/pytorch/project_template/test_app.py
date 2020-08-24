from fastapi.testclient import TestClient
import os
import json

from app import app

client = TestClient(app)

with open("config.json", "r") as file:
    config = json.load(file)
label = config["train"]["label"][0]

def test_unauthorized():
    response = client.post("/update-model")
    assert response.json() == {"error": "Unauthorized"}
    assert response.status_code == 401


def test_bad_request():
    response = client.post("/predict",
                           json={"wrong": "body"},
                           headers={"TOKEN": os.environ["API_TOKEN"]})
    assert response.json() == {"error": "Bad request"}
    assert response.status_code == 400


def test_update_features():
    response = client.post("/update-model", headers={"TOKEN": os.environ["API_TOKEN"]})
    assert response.json() == {"message": "model updated"}
    assert response.status_code == 200


def test_health_check():
    response = client.get("/")
    assert response.json() == {"message": "alive and running!"}
    assert response.status_code == 200


def test_predict():
    with open("request_model.json", "r") as file:
        request_model = json.load(file)

    response = client.post("/predict",
                           headers={"TOKEN": os.environ["API_TOKEN"]},
                           json=request_model)

    assert response.json().get(f"{label}") is not None
    assert response.status_code == 200
