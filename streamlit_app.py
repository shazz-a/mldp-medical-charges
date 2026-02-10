import joblib
import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Medical Insurance Charges Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Insurance Charges Prediction")
st.write(
    "This app predicts estimated medical insurance charges based on user profile inputs "
    "(age, BMI, smoker status, etc.)."
)


MODEL_PATH = "medical_charges_gbr_tuned.pkl"
model = joblib.load(MODEL_PATH)

st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
    sex = st.selectbox("Sex", ["female", "male"])

with col2:
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    bmi = st.slider("BMI", min_value=15.0, max_value=53.0, value=27.0, step=0.1)


errors = []
if bmi < 10 or bmi > 80:
    errors.append("BMI value looks unrealistic. Please enter a BMI within a reasonable range.")
if age < 18:
    errors.append("Age must be 18 or above.")
if children < 0:
    errors.append("Number of children cannot be negative.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

if st.button("Predict Charges"):
    df_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    pred = model.predict(df_input)[0]
    st.success(f"Predicted Medical Charges: ${pred:,.2f}")

    with st.expander("Show input summary"):
        st.dataframe(df_input)
