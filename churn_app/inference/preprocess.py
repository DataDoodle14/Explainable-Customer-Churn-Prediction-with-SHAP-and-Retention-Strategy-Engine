import joblib
import pandas as pd

preprocessor = joblib.load("churn_app/artifacts/preprocessor.pkl")

def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    return preprocessor.transform(df)
