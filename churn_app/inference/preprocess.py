import joblib
import pandas as pd

preprocessor = joblib.load("artifacts/preprocessor.pkl")

def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])
    return preprocessor.transform(df)
