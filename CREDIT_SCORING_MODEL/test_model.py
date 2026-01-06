import pickle
import joblib
import numpy as np
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

columns = joblib.load("columns.pkl")  

categorical_features = list(label_encoders.keys())

print("Please enter the following details:")

input_data = {}

for feature in columns:
    val = input(f"{feature}: ")

    if feature not in categorical_features:
        try:
            val = float(val)
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.")
            exit(1)

    input_data[feature] = val

features_list = []

for feature in columns:
    val = input_data[feature]

    if feature in categorical_features:
        le = label_encoders[feature]
        if val not in le.classes_:
            print(f"Invalid value '{val}' for {feature}. Valid options: {list(le.classes_)}")
            exit(1)
        val = le.transform([val])[0]

    features_list.append(val)

input_array = np.array(features_list).reshape(1, -1)

input_df = pd.DataFrame(input_array, columns=columns)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]

if prediction == 1:
    print("\nThe model predicts this person is likely to DEFAULT on the loan.")
else:
    print("\n The model predicts this person is likely to REPAY the loan successfully.")

