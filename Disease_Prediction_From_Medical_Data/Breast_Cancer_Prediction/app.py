
import streamlit as st
import numpy as np
import pickle
import joblib

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("🎗️ Breast Cancer Prediction System")
st.write("Enter tumor measurement details to predict breast cancer.")

st.markdown("---")

inputs = []
for col in columns:
    value = st.number_input(f"{col}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_data = np.array([inputs])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ Malignant Tumor Detected (Breast Cancer Positive).")
    else:
        st.success("✅ Benign Tumor (No Breast Cancer).")
