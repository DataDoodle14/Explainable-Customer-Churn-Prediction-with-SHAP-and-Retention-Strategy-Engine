
import streamlit as st

from ui.input_form import get_user_input
from inference.preprocess import preprocess_input
from inference.predict import predict_churn
from explainability.shap_explainer import compute_shap, plot_waterfall
from explainability.retention_logic import generate_retention_plan


st.set_page_config(page_title="Churn XAI", layout="wide")

st.title("Customer Churn Prediction with Explainable AI")
st.caption("Predict churn risk and understand key drivers using machine learning and SHAP.")


# ───────────────── Sidebar ─────────────────
with st.sidebar:
    st.header("Customer Details")
    user_input = get_user_input()
    predict_btn = st.button("Predict Churn Risk")


# ───────────────── Main logic ─────────────────
if predict_btn:
    # 1️⃣ Preprocess
    X_transformed = preprocess_input(user_input)

    # 2️⃣ Predict
    prob, pred = predict_churn(X_transformed)

    st.subheader("Prediction Summary")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
    with col2:
        if pred:
               st.markdown(
               "<div style='background-color:#fdecea;color:#b71c1c;"
               "padding:10px 14px;border-radius:8px;font-weight:600;'>"
               "⚠ HIGH CHURN RISK</div>",
               unsafe_allow_html=True
               )
        else:
               st.markdown(
               "<div style='background-color:#e8f5e9;color:#1b5e20;"
               "padding:10px 14px;border-radius:8px;font-weight:600;'>"
               "✔ LOW CHURN RISK</div>",
               unsafe_allow_html=True
              )

    

    st.caption("Decision threshold: 0.30")

    # 3️⃣ SHAP (SAFE here)
    shap_vals = compute_shap(X_transformed)

    # 4️⃣ Tabs
    tab_explain, tab_actions = st.tabs(["Explanation", "Retention Actions"])

    # ───────── Explanation Tab ─────────
    with tab_explain:
        st.subheader("Why did the model make this prediction?")

        fig = plot_waterfall(shap_vals, X_transformed)
        st.pyplot(fig, width="stretch")

        st.markdown(
            "The model's decision is influenced by a combination of customer engagement, "
            "account activity, and product ownership factors."
        )

    # ───────── Retention Tab ─────────
    with tab_actions:
        if pred:
            st.subheader("Suggested Retention Actions")
            st.caption(
                "These recommendations are derived from model explanations and are intended "
                "as decision support."
            )

            actions = generate_retention_plan(
                shap_vals[0],
                X_row=X_transformed[0]
            )

            if actions:
                for a in actions:
                    st.write("•", a)
            else:
                st.info("No dominant risk drivers identified for targeted intervention.")
        else:
            st.info("Customer is currently low risk. No immediate retention action required.")


st.divider()
st.caption("Model: LightGBM | Explainability: SHAP | Threshold: 0.30")
