import shap
import joblib
import matplotlib.pyplot as plt

model = joblib.load("artifacts/model.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")

explainer = shap.TreeExplainer(model)

def compute_shap(X_transformed):
    shap_values = explainer.shap_values(X_transformed)
    return shap_values[1] if isinstance(shap_values, list) else shap_values

def plot_waterfall(shap_vals, X_row):
    fig = plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_row[0],
            feature_names=feature_names
        ),
        show=False
    )
    return fig
