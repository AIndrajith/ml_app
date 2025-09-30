import streamlit as st
import joblib

# Load the model and scaler
model = joblib.load("linear_regressio_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app
st.title("MetaBrains student Test Score Predictor")
st.write("Enter the number of hours studied to predict the test score.")

# User Input 
hours = st.number_input("Hours studied: ", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write(f"Predict Test Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")