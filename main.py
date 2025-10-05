import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open('diabetes_model.pkl', 'rb'))
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health details below to check if you might have diabetes.")

# Input fields for user
pregnancies = st.number_input("pregnancy Level", min_value=0, max_value=17, value=12)
glucose = st.number_input("Glucose Level", min_value=0, max_value=199, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.5)
age = st.number_input("Age", min_value=0, max_value=81, value=30)

# Predict button
if st.button("Check Diabetes"):
    # prepare data for prediction
    input_data = np.array([[pregnancies,glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have Diabetes.")
    else:
        st.success("✅ The person is not likely to have Diabetes.")
