"""Trains the house price model and saves it to submission/house_predictor.pkl."""

import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "house_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "submission", "house_predictor.pkl")

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    return df[FEATURES], df["price"]


def train(features, target):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"R2 (test):  {r2_score(y_test, y_pred):.3f}")
    print(f"MAE (test): {mean_absolute_error(y_test, y_pred):,.0f}")

    model.fit(features, target)
    return model


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    save_model(train(*load_data()))