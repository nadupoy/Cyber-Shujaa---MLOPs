# Tasks 9 & 10: Create a simple FastAPI app with a /predict endpoint
# Load the .joblib model and return predictions for incoming JSON input

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# --- 1. MOCK MODEL LOADING ---
# NOTE: In a real environment, you must have 'california_housing_knr_pipeline.joblib' in the same directory as this app.py file.
try:
  # Load the actual trained pipeline object saved in Task 7
  MODEL_PATH = 'california_housing_knr_pipeline.joblib'
  if not os.path.exists(MODEL_PATH):
    # Placeholder for demonstration if the file is missing
    print(f"WARNING: {MODEL_PATH} not found. Using mock model for demonstration.")

    # Define a mock class to simulate the scikit-learn pipeline's predict method
    class MockPipeline:
      def predict(self, df):
        # Simple mock prediction logic: return a value based on median income
        # This ensures the app runs even if the .joblib file isn't present
        predictions = df['MedInc'] * 0.5 + 1.0
        return predictions.tolist()

    pipeline = MockPipeline()
  else:
    # Load the actual pipeline if the file exists
    pipeline = joblib.load(MODEL_PATH)
    print(f"SUCCESS: Loaded model from {MODEL_PATH}")

except Exception as e:
  print(f"Error loading model: {e}")
  # Exit or use mock pipeline if critical loading fails

# --- 2. Pydantic Data Model ---
# This defines the required input structure for the /predict endpoint.
# FastAPI uses this for automatic validation and documentation.
class HousingFeatures(BaseModel):
  """Defines the structure of the 8 features expected by the model."""
  MedInc: float
  HouseAge: float
  AveRooms: float
  AveBedrms: float
  Population: float
  AveOccup: float
  Latitude: float
  Longitude: float

# --- 3. FastAPI Application Setup ---
app = FastAPI(
  title="California Housing Price Predictor (CyberShujaa MLOps)",
  description="A simple API to predict median house value using a KNR pipeline",
  version="1.0.0"
)

# --- 4. Prediction Endpoint (/predict) ---
@app.post("/predict")
def predict_house_price(data: HousingFeatures):
  """
  Recieves JSON data, converts it to a DataFrame, and returns the prediction.
  """
  try:
    # Convert the Pydantic model data into a Pandas DataFrame
    # The key is to convert it to a dictionary and then wrap it in a list to ensure it has one row, which the pipeline expects.
    input_data = data.model_dump()
    input_df = pd.DataFrame([input_data])

    # Make the prediction using the loaded pipeline (which handles scaling internally)
    prediction_array = pipeline.predict(input_df)

    # The prediction is in $100,000s, so we format it for readability
    prediction_value = float(prediction_array[0])

    # Return the result
    return {
      "predicted_median_house_value": prediction_value,
      "unit": "in $100,000s"
    }

  except Exception as e:
    return {"error": str(e)}

# --- 5. Simple Root Endpoint (Optional check) ---
@app.get("/")
def read_root():
  return {
    "message": "Welcome to the California Housing Price Prediction API. Use the /predict endpoint (POST) or visit /docs."
  }
