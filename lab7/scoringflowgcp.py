from metaflow import FlowSpec, step, Parameter, kubernetes, retry, timeout, catch, conda, conda_base
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib

@conda_base(python="3.8.12", libraries={'scikit-learn': '1.0.2', 'pandas': '1.3.5', 
                                      'numpy': '1.21.5', 'mlflow': '1.24.0',
                                      'joblib': '1.1.0'})
class ScoringFlow(FlowSpec):
    """A flow for making predictions using the trained Airbnb price prediction model"""
    
    # Parameters
    test_features_path = Parameter('test_features', 
                               help='Path to test features for scoring',
                               default='../data/processed_test_features.csv')
    test_target_path = Parameter('test_target',
                              help='Path to test target for evaluation',
                              default='../data/processed_test_target.csv')
    scaler_path = Parameter('scaler_path',
                           help='Path to the saved feature scaler',
                           default='../data/feature_scaler.joblib')
    model_stage = Parameter('model_stage',
                          help='Stage of the model to use (None, Staging, Production)',
                          default='None')
    mlflow_tracking_uri = Parameter('mlflow_uri', 
                                  help='URI for MLflow tracking server',
                                  default='http://127.0.0.1:5000/') #TODO: change to GCP

    @step
    @retry(times=3)
    def start(self):
        """Start the flow and load the data"""
        print("Loading test data...")
        # Load test features and targets, explicitly handling headers
        self.features = pd.read_csv(self.test_features_path, header=0)  # header=0 means first row is header
        self.targets = pd.read_csv(self.test_target_path, header=0)    # header=0 means first row is header
        
        print(f"Loaded {len(self.features)} feature rows")
        print(f"Loaded {len(self.targets)} target rows")
        
        # Check for size mismatch early
        if len(self.features) != len(self.targets):
            print(f"WARNING: Size mismatch between features ({len(self.features)}) and targets ({len(self.targets)})")
            print("This may indicate an issue with the data files or headers")
            
        self.next(self.prepare_features)

    @step
    @retry(times=3)
    def prepare_features(self):
        """Prepare features for scoring using same preprocessing as training"""
        print("Preparing features...")
        
        # Keep only numeric features and handle missing values
        self.X = self.features.select_dtypes(include=[np.number])
        self.X = self.X.fillna(self.X.mean())
        
        # Use the same scaler that was used during training
        try:
            print(f"Loading scaler from {self.scaler_path}")
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.X_scaled = self.scaler.transform(self.X)
                print("Successfully loaded and applied the training scaler")
            else:
                # If scaler file is not found, fall back to fitting a new scaler
                print(f"Warning: Scaler file not found at {self.scaler_path}")
                print("Fitting a new scaler (this may cause inconsistent preprocessing)")
                self.scaler = StandardScaler()
                self.X_scaled = self.scaler.fit_transform(self.X)
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            print("Falling back to fitting a new scaler (this may cause inconsistent preprocessing)")
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Extract target and ensure consistent dimensions
        self.y_true = self.targets['price'].values
        
        # Handle any remaining size mismatch
        if len(self.X_scaled) != len(self.y_true):
            print(f"WARNING: Size mismatch after preprocessing - X_scaled: {len(self.X_scaled)}, y_true: {len(self.y_true)}")
            min_size = min(len(self.X_scaled), len(self.y_true))
            self.X_scaled = self.X_scaled[:min_size]
            self.y_true = self.y_true[:min_size]
            print(f"Adjusted to minimum size: {min_size} rows")
        
        self.next(self.load_model)

    @step
    @retry(times=3)
    @timeout(minutes=10)
    def load_model(self):
        """Load the registered model from MLFlow"""
        print("Loading model...")
        
        # Set up MLFlow tracking
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Load the registered model
        client = MlflowClient()
        model_name = "airbnb_price_predictor"
        
        # Get the latest version or specific stage
        if self.model_stage == 'None':
            model_version = client.get_latest_versions(model_name, stages=['None'])[0]
        else:
            model_version = client.get_latest_versions(model_name, stages=[self.model_stage])[0]
            
        # Load the model from the specified version
        model_uri = f"models:/{model_name}/{model_version.version}"
        print(f"Loading model from {model_uri}")
        self.model = mlflow.sklearn.load_model(model_uri)
        
        self.next(self.make_predictions)

    @kubernetes
    @step
    @timeout(minutes=15)
    @retry(times=2)
    @catch(var='prediction_error')
    def make_predictions(self):
        """Make predictions using the loaded model"""
        print("Making predictions...")
        
        if hasattr(self, 'prediction_error'):
            print(f"Caught an error during predictions: {self.prediction_error}")
            raise RuntimeError("Prediction failed after retries")
            
        # Generate predictions
        self.predictions = self.model.predict(self.X_scaled)
        
        print(f"Features shape: {self.X_scaled.shape}")
        print(f"Targets shape: {self.y_true.shape}")
        print(f"Predictions shape: {self.predictions.shape}")
        
        # Create a DataFrame with predictions and actual values
        self.predictions_df = pd.DataFrame({
            'actual_price': self.y_true,
            'predicted_price': self.predictions,
            'error': self.y_true - self.predictions
        })
            
        self.next(self.evaluate_performance)
        
    @step
    @retry(times=2)
    def evaluate_performance(self):
        """Evaluate model performance on test data"""
        print("Evaluating model performance...")
        
        # Calculate metrics
        self.rmse = np.sqrt(mean_squared_error(self.y_true, self.predictions))
        self.r2 = r2_score(self.y_true, self.predictions)
        
        # Print performance metrics
        print(f"\nTest Set Performance:")
        print(f"RMSE: ${self.rmse:.2f}")
        print(f"R² Score: {self.r2:.3f}")
        
        # Calculate additional error statistics
        self.mean_abs_error = np.mean(np.abs(self.predictions_df['error']))
        self.mean_pct_error = np.mean(np.abs(self.predictions_df['error'] / self.predictions_df['actual_price']))
        
        print(f"Mean Absolute Error: ${self.mean_abs_error:.2f}")
        print(f"Mean Percentage Error: {self.mean_pct_error*100:.1f}%")
        
        self.next(self.save_predictions)

    @step
    @retry(times=3)
    def save_predictions(self):
        """Save predictions to a file"""
        print("Saving predictions...")
        
        # Save predictions to CSV
        output_path = 'test_predictions.csv'
        self.predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Print some basic statistics
        print("\nPrediction Statistics:")
        print(f"Mean predicted price: ${self.predictions.mean():.2f}")
        print(f"Mean actual price: ${self.y_true.mean():.2f}")
        print(f"Min predicted price: ${self.predictions.min():.2f}")
        print(f"Max predicted price: ${self.predictions.max():.2f}")
        
        self.next(self.end)

    @step
    def end(self):
        """End the flow"""
        print("Flow completed!")
        print(f"Made predictions for {len(self.predictions)} test instances")
        print(f"Final RMSE on test set: ${self.rmse:.2f}")
        print(f"Final R² on test set: {self.r2:.3f}")

if __name__ == '__main__':
    ScoringFlow()
