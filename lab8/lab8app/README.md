# Simulated Data Classification API

This is a FastAPI application that uses a machine learning model from MLflow to predict classifications for simulated data.

## Setup

1. Make sure you have the required dependencies:
   ```
   pip install fastapi uvicorn mlflow pydantic
   ```

2. Ensure your MLflow model is registered:
   - The app looks for a model named "simulated_data_classification_model_1" in the MLflow model registry

## Running the API

1. Start the FastAPI server:
   ```
   python app.py
   ```

2. The server will start at http://localhost:8001

## API Endpoints

### Root Endpoint
- URL: GET http://localhost:8001/
- Returns a welcome message and basic information about the API

### Prediction Endpoint
- URL: POST http://localhost:8001/predict
- Accepts a JSON body with a "features" field containing an array of numerical values
- Returns predictions (probability scores for each class)

## Usage Examples

### Example (curl):
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.5, 0.2, 0.3, 0.8]}'
```

### Testing Script
Run the provided test script to check if the API is working correctly:
```bash
python test_app.py
```
This will send sample feature sets to the API and display the results. 