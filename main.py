from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file in the same directory
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        # Use environment variable to load model and scaler paths
        model_path = os.getenv("MODEL_PATH", "linear_regressio_model.pkl")
        scaler_model = os.getenv("SCALER_PATH", "scaler.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_model)

        logging.info("Model and scaler have been loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        raise HTTPException(status_code=500, detail="Error loading model")

# define a request model for the input
class PredictionRequest(BaseModel):
    hours_studied: float

# Home endpoint
@app.get("/")
def home():
    return{
        "message":"Welcome to Meta Brain's Prediction Model"
    }

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    
    if model is None or scaler is None:
        logging.error("Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded, Please try again later")
    
    # Input validation: Ensure hours_studied should be positive
    if request.hours_studied <= 0:
        logging.warning("Received invalid input: Hours studied should be positive.")
        raise HTTPException(status_code=400, detail="Hours studied should be a positive number.")
    
    hours = request.hours_studied       # Extract the hours studied from the request
    data = pd.DataFrame([[hours]], columns=['Hours Studied'])   # Prepare the data for prediction
    scaled_data = scaler.transform(data)        # Scale the input data

    try:
        prediction = model.predict(scaled_data)     # Make prediction using the model
        logging.info(f"Prediction for {hours} hours: {prediction[0]}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

    return {
        "predicted_test_score":prediction[0]    # Return the predicted test score
    }