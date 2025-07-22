
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("enhanced_salary_model.pkl")

# App Title
st.title("Employee Salary Prediction App")
st.subheader("Using Linear Regression - IBM Capstone Project")

# Sidebar Inputs
st.sidebar.header("Input Employee Details")

years_exp = st.sidebar.slider("Years of Experience", 0.0, 15.0, 3.0, step=0.5)

# Predict Button
if st.sidebar.button("Predict Salary"):
    prediction = model.predict([[years_exp]])
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")

# Info Section
st.markdown("---")
st.markdown("**Model Info:**")
st.write("- Trained using Linear & Ridge Regression")
st.write("- Best Model Chosen by R² Score (Linear Regression)")
st.write("- Dataset: 35 samples of Experience vs Salary")
st.write("- Created by **Kasara Pavan Sai Reddy**")

# Footer
st.markdown("---")
st.caption("IBM x Edunet Foundation | Capstone Internship 2025")
