import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load the Airbnb dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Handle missing values
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    df['overall_satisfaction'] = df['overall_satisfaction'].fillna(0)  # 0 for no reviews
    
    # Encode categorical variables
    le = LabelEncoder()
    df['room_type'] = le.fit_transform(df['room_type'])
    df['neighborhood'] = le.fit_transform(df['neighborhood'])
    
    # Create features and target
    features = ['room_type', 'neighborhood', 'reviews', 'overall_satisfaction', 
               'accommodates', 'bedrooms']
    X = df[features]
    y = df['price']
    
    return X, y

def main():
    # Load data
    df = load_data('data/airbnb.csv')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed datasets
    pd.DataFrame(X_train).to_csv('data/processed_train_features.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed_test_features.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/processed_train_target.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/processed_test_target.csv', index=False)

if __name__ == "__main__":
    main()