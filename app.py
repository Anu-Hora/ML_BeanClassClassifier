import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# Load Model & Scaler
# =========================
model = joblib.load(open("BeanClassifier.pkl", "rb"))
scaler = joblib.load(open("BeanClassifierScaler.pkl", "rb"))

st.title("🌱 Dry Bean Classification App")
st.write("Enter bean features to predict class")

# =========================
# Input Fields
# =========================

area = st.number_input("Area")
perimeter = st.number_input("Perimeter")
major_axis = st.number_input("Major Axis Length")
minor_axis = st.number_input("Minor Axis Length")
aspect_ratio = st.number_input("Aspect Ratio")
eccentricity = st.number_input("Eccentricity")
convex_area = st.number_input("Convex Area")
equiv_diameter = st.number_input("Equivalent Diameter")
extent = st.number_input("Extent")
solidity = st.number_input("Solidity")
roundness = st.number_input("Roundness")
compactness = st.number_input("Compactness")
sf1 = st.number_input("ShapeFactor1")
sf2 = st.number_input("ShapeFactor2")
sf3 = st.number_input("ShapeFactor3")
sf4 = st.number_input("ShapeFactor4")

# =========================
# Prediction
# =========================

if st.button("Predict Bean Type"):
    
    # Create input array
    input_data = np.array([[
        area, perimeter, major_axis, minor_axis, aspect_ratio,
        eccentricity, convex_area, equiv_diameter, extent,
        solidity, roundness, compactness, sf1, sf2, sf3, sf4
    ]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)

    # If you used LabelEncoder
    class_names = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMOSAN', 'HOROZ', 'SIRA']
    
    st.success(f"🌾 Predicted Bean Type: {class_names[prediction[0]]}")