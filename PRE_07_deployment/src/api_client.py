"""Client for the prediction API (api_server.py must be running)."""

import os

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5001/predict")

house = {
    "bedrooms": 3,
    "bathrooms": 1,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1,
    "waterfront": 0,
    "condition": 3,
}


def get_prediction(data):
    response = requests.post(API_URL, json=data, timeout=10)
    response.raise_for_status()
    return response.json()["prediction"]


if __name__ == "__main__":
    print(f"Estimated price: {get_prediction(house):,.2f}")