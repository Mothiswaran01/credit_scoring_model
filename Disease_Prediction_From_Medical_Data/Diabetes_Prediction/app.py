
import streamlit as st
import numpy as np
import pickle
import joblib

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict the possibility of diabetes.")

st.markdown("---")

Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array([[ 
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")
