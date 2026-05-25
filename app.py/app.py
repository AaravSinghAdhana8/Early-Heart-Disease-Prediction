from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = [
    PROJECT_ROOT / "model.pkl",
    PROJECT_ROOT / "heart.csv" / "model.pkl",
]

MODEL_PATH = next((path for path in MODEL_PATHS if path.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find model.pkl. Run model_training.py first or place the model file in the project root."
    )

# Load saved model
model = joblib.load(MODEL_PATH)

# Title
st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below")

# User Inputs

age = st.number_input("Age", 1, 100, 45)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    200,
    120
)

chol = st.number_input(
    "Cholesterol Level",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

restecg = st.selectbox(
    "Rest ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    60,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3],
    index=3
)

# Convert sex to numeric
sex_value = 1 if sex == "Male" else 0

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame([
        {
            "age": age,
            "sex": sex_value,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Person is at risk of Heart Disease")
    else:
        st.success("✅ Person is NOT at risk of Heart Disease")