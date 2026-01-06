
import streamlit as st
import numpy as np
import pickle
import joblib

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict the possibility of heart disease.")

st.markdown("---")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[ 
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Heart Disease.")
    else:
        st.success("✅ The patient is unlikely to have Heart Disease.")
