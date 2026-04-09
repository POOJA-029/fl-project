import streamlit as st
import numpy as np
from model import create_model

model = create_model()

st.title("Diabetes Prediction System")

features = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]

inputs = []

for f in features:
    inputs.append(st.number_input(f))

if st.button("Predict"):
    data = np.array([inputs])
    pred = model.predict(data)[0][0]

    if pred > 0.5:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")