import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(input_file, model_file):
    df = pd.read_csv(input_file)
    
    # Prepare data
    X = df.drop('Demand_Forecast', axis=1)
    y = df['Demand_Forecast']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae:.2f}")
    
    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model('processed_data.csv', 'demand_forecast_model.pkl')