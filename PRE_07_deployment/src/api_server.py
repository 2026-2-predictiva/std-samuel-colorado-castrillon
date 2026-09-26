"""REST API that serves house price predictions.

Run:  python api_server.py
POST /predict with a JSON body containing the model features.
"""

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

from train_model import FEATURES, MODEL_FILE

app = Flask(__name__)

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    features = pd.DataFrame([{feature: float(payload[feature]) for feature in FEATURES}])
    prediction = model.predict(features)[0]
    return jsonify({"prediction": round(float(prediction), 2)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)