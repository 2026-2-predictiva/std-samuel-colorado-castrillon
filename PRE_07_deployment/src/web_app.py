"""Web application that estimates house prices using templates/index.html.

Run:  python web_app.py  and open http://127.0.0.1:5000
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

from train_model import FEATURES, MODEL_FILE

app = Flask(__name__)

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


def form_to_features(form):
    values = {
        "bedrooms": float(form["bedrooms"]),
        "bathrooms": float(form["bathrooms"]),
        "sqft_living": float(form["sqft_living"]),
        "sqft_lot": float(form["sqft_lot"]),
        "floors": float(form["floors"]),
        "waterfront": 1 if form.get("waterfront") == "Yes" else 0,
        "condition": int(form["condition"]),
    }
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            price = model.predict(form_to_features(request.form))[0]
            prediction = f"$ {price:,.2f}"
        except (KeyError, ValueError):
            prediction = "Invalid input, please check the values."
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)