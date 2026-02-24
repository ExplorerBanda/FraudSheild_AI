import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open('model.pkl', 'rb') as modal_file:
    model = pickle.load(modal_file)

# Streamlit app title and instructions
st.title("FraudShield AI - Transaction Fraud Detection")
st.write("""
    This web app predicts whether a transaction is fraudulent or genuine based on the transaction details.
""")

# Input fields for user to fill out (you can customize based on your dataset columns)
transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.1)
user_id = st.number_input("User ID", min_value=0)
transaction_type = st.selectbox("Transaction Type", ['purchase', 'refund', 'withdrawal'])
merchant_id = st.text_input("Merchant ID", "")
device_id = st.text_input("Device ID", "")

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Process the input data into a numpy array that matches the model’s expected input
    input_data = np.array([[transaction_amount, user_id, transaction_type, merchant_id, device_id]])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)

    # Output the prediction result
    if prediction[0] == 1:
        st.write(f"Prediction: **Fraudulent**")
    else:
        st.write(f"Prediction: **Genuine**")

    # Display confidence level
    st.write(f"Confidence Level: {prediction_prob[0][1]:.2f}")
