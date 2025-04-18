from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow.tracking import MlflowClient
import os
import joblib

class TrainingFlow(FlowSpec):
    """A flow for training and registering a model for Airbnb price prediction"""
    
    # Parameters that can be passed to the flow
    random_seed = Parameter('seed', help='Random seed for reproducibility', default=42)
    n_estimators = Parameter('n_estimators', help='Number of trees in random forest', default=100)
    test_size = Parameter('test_size', help='Size of test split', default=0.2)
    cv_folds = Parameter('cv_folds', help='Number of cross validation folds', default=5)
    output_dir = Parameter('output_dir', help='Directory to save test data and model artifacts', default='../data')

    @step
    def start(self):
        """Start the flow and load the data"""
        print("Loading data...")
        # Load Airbnb dataset
        self.data = pd.read_csv('../data/airbnb.csv')
        self.next(self.prepare_features)

    @step
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Separate features and target
        self.X = self.data.drop('price', axis=1)
        self.y = self.data['price']
        
        # Handle missing values
        self.X = self.X.select_dtypes(include=[np.number])  # Keep only numeric columns for simplicity
        self.X = self.X.fillna(self.X.mean())
        
        # Keep original data for saving test set later
        self.X_original = self.X.copy()
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = self.X.columns.tolist()
        
        # Split data - use train_test_split to get indices for both X and y
        X_train_idx, X_test_idx = train_test_split(
            np.arange(len(self.X)),
            test_size=self.test_size,
            random_state=self.random_seed
        )
        
        # Use the indices to split both X and y
        self.X_train = self.X_scaled[X_train_idx]
        self.X_test = self.X_scaled[X_test_idx]
        self.y_train = self.y.iloc[X_train_idx].values
        self.y_test = self.y.iloc[X_test_idx].values
        
        # Save test indices for later use
        self.test_indices = X_test_idx
        
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model with cross validation"""
        print("Training model...")
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_seed
        )
        
        # Perform cross validation
        self.cv_scores = cross_val_score(
            self.model,
            self.X_train,
            self.y_train,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error'
        )
        
        # Train final model on full training data
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on the test set
        self.test_predictions = self.model.predict(self.X_test)
        self.test_rmse = np.sqrt(np.mean((self.test_predictions - self.y_test) ** 2))
        
        self.next(self.save_test_data)
        
    @step
    def save_test_data(self):
        """Save test data for later scoring"""
        print("Saving test data for later scoring...")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Use the saved test indices to extract original features
        test_features = self.X_original.iloc[self.test_indices]
        test_targets = pd.DataFrame({'price': self.y.iloc[self.test_indices]})
        
        # Verify they have the same length
        print(f"Test features shape: {test_features.shape}")
        print(f"Test targets shape: {test_targets.shape}")
        
        if len(test_features) != len(test_targets):
            print("ERROR: Test features and targets have different lengths!")
            return
            
        # Save test features and targets
        test_features_path = os.path.join(self.output_dir, "processed_test_features.csv")
        test_features.to_csv(test_features_path, index=False)
        print(f"Saved test features to {test_features_path}")
        
        test_target_path = os.path.join(self.output_dir, "processed_test_target.csv")
        test_targets.to_csv(test_target_path, index=False)
        print(f"Saved test target to {test_target_path}")
        
        # Save the scaler for consistent preprocessing
        scaler_path = os.path.join(self.output_dir, "feature_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model with MLFlow"""
        print("Registering model...")
        
        # Set up MLFlow tracking
        mlflow.set_tracking_uri('http://127.0.0.1:5000/')
        mlflow.set_experiment('airbnb_price_prediction')
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param('n_estimators', self.n_estimators)
            mlflow.log_param('random_seed', self.random_seed)
            mlflow.log_param('cv_folds', self.cv_folds)
            
            # Log metrics
            cv_rmse = np.sqrt(-self.cv_scores.mean())
            mlflow.log_metric('cv_rmse', cv_rmse)
            mlflow.log_metric('test_rmse', self.test_rmse)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "random_forest_model",
                registered_model_name="airbnb_price_predictor"
            )
            
            # Log feature names and preprocessing artifacts
            mlflow.log_param('features', self.feature_names)
            
            # Log scaler as an artifact
            mlflow.log_artifact(os.path.join(self.output_dir, "feature_scaler.joblib"))
        
        self.next(self.end)

    @step
    def end(self):
        """End the flow"""
        print("Flow completed!")
        print(f"Cross validation RMSE: {np.sqrt(-self.cv_scores.mean()):.2f}")
        print(f"Test set RMSE: {self.test_rmse:.2f}")

if __name__ == '__main__':
    TrainingFlow()