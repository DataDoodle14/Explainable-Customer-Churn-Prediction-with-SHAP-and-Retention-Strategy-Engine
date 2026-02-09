import joblib

# Load artifacts once
FEATURE_NAMES = joblib.load("artifacts/feature_names.pkl")

RETENTION_ACTIONS = {
    "isactivemember": {
        "condition": lambda x: x == 0,
        "message": "Customer is inactive. Consider engagement campaigns or loyalty rewards."
    },
    "numofproducts": {
        "condition": lambda x: x == 1,
        "message": "Customer has only one product. Cross-sell additional services."
    },
    "has_balance": {
        "condition": lambda x: x == 1,
        "message": "Customer maintains a balance. Consider personalized financial offers."
    },
    "age": {
        "condition": lambda x: x > 45,
        "message": "Older customer segment detected. Offer long-term relationship benefits."
    }
}


def generate_retention_plan(shap_row, X_row, min_shap=0.05):
    actions = []

    for feature, shap_val, value in zip(FEATURE_NAMES, shap_row, X_row):
        if shap_val > min_shap and feature in RETENTION_ACTIONS:
            rule = RETENTION_ACTIONS[feature]
            if rule["condition"](value):
                actions.append(rule["message"])

    return actions
