import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
model = joblib.load("final_rf_log_tuned_pipeline.joblib")

st.set_page_config(page_title="Medical Charges Predictor", layout="centered")
st.title("💉 Medical Charges Prediction App")
st.write("Enter patient details to estimate annual medical charges.")

# User inputs
age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Create input DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# Predict button
if st.button("Predict Charges"):
    log_pred = model.predict(input_df)
    pred = np.expm1(log_pred)
    st.success(f"💰 Predicted Medical Charges: ${pred[0]:,.2f}")
