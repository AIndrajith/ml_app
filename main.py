from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the saved model ans scaler
model = None
scaler = None

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, scaler
    model = joblib.load('linear_regressio_model.pkl')
    scaler = joblib.load('scaler.pkl')

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
    hours = request.hours_studied       # Extract the hours studied from the request
    data = pd.DataFrame([[hours]], columns=['Hours Studied'])   # Prepare the data for prediction
    scaled_data = scaler.transform(data)        # Scale the input data
    prediction = model.predict(scaled_data)     # Make prediction using the model
    return {
        "predicted_test_score":prediction[0]    # Return the predicted test score
    }