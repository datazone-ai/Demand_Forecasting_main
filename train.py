import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(input_path, model_output_path):
    # Load preprocessed data
    df = pd.read_csv(input_path)
    
    # Define features and target variable
    X = df.drop(columns=['Demand Forecast'])  # Features
    y = df['Demand Forecast']  # Target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Manually computing RMSE

    print(f'Model Evaluation:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model training complete. Model saved to {model_output_path}")


# Run training
if __name__ == "__main__":
    train_model("processed_data/medicine_inventory_cleaned.csv", "models/demand_forecasting_rf.pkl")

