import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
columns = joblib.load("columns.pkl")

categorical_features = list(label_encoders.keys())

st.title("Credit Default Prediction")

user_input = {}

for feature in columns:
    if feature in categorical_features:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    else:
        user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict"):
    input_list = []
    for feature in columns:
        val = user_input[feature]
        if feature in categorical_features:
            le = label_encoders[feature]
            val = le.transform([val])[0]
        input_list.append(val)

    input_array = np.array(input_list).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("This person is likely to DEFAULT on the loan.")
    else:
        st.success("This person is likely to REPAY the loan successfully.")
