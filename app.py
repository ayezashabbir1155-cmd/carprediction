
import streamlit as st
import pickle
import pandas as pd

# ============================================
# LOAD MODEL
# ============================================

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================
# LOAD SCALER
# ============================================

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ============================================
# LOAD COLUMNS
# ============================================

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ============================================
# STREAMLIT UI
# ============================================

st.title("🚗 Car Price Prediction App")

st.write("Enter car details below:")

# ============================================
# USER INPUTS
# ============================================

year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025)

present_price = st.number_input("Present Price (in lakhs)", min_value=0.0)

kms_driven = st.number_input("Kilometers Driven", min_value=0)

owner = st.number_input("Number of Previous Owners", min_value=0, max_value=5)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# ============================================
# CREATE INPUT DATAFRAME
# ============================================

input_data = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner],
    'Fuel_Type': [fuel_type],
    'Seller_Type': [seller_type],
    'Transmission': [transmission]
})

# ============================================
# ONE HOT ENCODING
# ============================================

input_data = pd.get_dummies(input_data)

# Add missing columns
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Keep same column order
input_data = input_data[columns]

# ============================================
# SCALE INPUT
# ============================================

input_scaled = scaler.transform(input_data)

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Car Price"):

    prediction = model.predict(input_scaled)

    st.success(f"🚘 Predicted Car Selling Price: {prediction[0]:,.2f} Lakhs")
