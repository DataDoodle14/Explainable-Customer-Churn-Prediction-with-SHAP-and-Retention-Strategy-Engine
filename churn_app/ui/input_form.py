import streamlit as st

def get_user_input():
    return {
        "creditscore": st.number_input("Credit Score", 300, 900),
        "age": st.number_input("Age", 18, 100),
        "balance": st.number_input("Balance", 0.0),
        "numofproducts": st.selectbox("Number of Products", [1,2,3,4]),
        "isactivemember": st.selectbox("Is Active Member", [0,1]),
        "gender": st.selectbox("Gender", ["Male","Female"]),
        "geography": st.selectbox("Geography", ["France","Germany","Spain"]),
        "estimatedsalary": st.number_input("Estimated Salary", 0.0),
        "hascrcard": st.selectbox("Has Credit Card", [0,1]),
        "tenure": st.number_input("Tenure", 0, 10),
        "has_balance": st.selectbox("Has Balance", [0,1])
    }
