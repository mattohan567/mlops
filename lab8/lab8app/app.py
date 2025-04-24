from fastapi import FastAPI
import mlflow.pyfunc
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Classification API",
    description="Classify simulated data",
    version="0.1"
)

class request_body(BaseModel):
    features: list

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    model_pipeline = mlflow.pyfunc.load_model("models:/simulated_data_classification_model_1/latest")

@app.get('/')
def main():
    return {'message': 'API for classifying simulated data'}

@app.post('/predict')
def predict(data: request_body):
    predictions = model_pipeline.predict([data.features])
    return {'predictions': predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
