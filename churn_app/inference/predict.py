import joblib
import numpy as np

model = joblib.load("churn_app/artifacts/model.pkl")

THRESHOLD = 0.3

def predict_churn(X_transformed):
    prob = model.predict_proba(X_transformed)[0, 1]
    pred = int(prob >= THRESHOLD)
    return prob, pred
