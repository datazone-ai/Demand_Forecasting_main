import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# Load preprocessed data
df = pd.read_csv("processed_data/medicine_inventory_cleaned.csv")

# Load trained model
model = joblib.load("models/demand_forecasting_rf.pkl")

# Define features (excluding target column)
features = [
    'SKU', 'Stock In', 'Stock Out', 'Stock Balance', 'Days_Until_Expiry', 'Lead_Time',
    'Season', 'Holiday', 'Promotion', 'Lag_1', 'Lag_2', 'Lag_3', 
    'Moving Average', 'Order_Day', 'Order_Month', 'Order_Weekday', 'Days_Since_Last_Order'
]

# Predict demand for each SKU
df['Predicted Demand'] = model.predict(df[features])

# Generate forecasts for May (Next Month) and June (Next+1 Month)
forecast_months = [1, 2]  # Forecast for the next 2 months

all_forecasts = []

for month_offset in forecast_months:
    forecast_date = pd.to_datetime('today') + pd.DateOffset(months=month_offset)
    forecast_month = forecast_date.month
    forecast_year = forecast_date.year

    df['Order Month'] = forecast_month
    df['Order Year'] = forecast_year  # Ensure correct year handling

    monthly_forecast = df.groupby(['SKU', 'Order Month', 'Order Year']).agg({
        'Predicted Demand': 'sum',  # Sum demand at monthly level
        'Stock Balance': 'last'  # Latest stock balance
    }).reset_index()

    # Define reorder threshold
    reorder_threshold = 50  

    # Determine if an order is needed
    monthly_forecast['Reorder Needed'] = monthly_forecast['Stock Balance'] < reorder_threshold

    # Calculate reorder quantity
    monthly_forecast['Reorder Quantity'] = np.where(
        monthly_forecast['Reorder Needed'], 
        np.maximum(0, monthly_forecast['Predicted Demand'] - monthly_forecast['Stock Balance']),
        0
    )

    # Set recommended order date (custom placeholder when no order is needed)
    monthly_forecast['Recommended Order Date'] = np.where(
        monthly_forecast['Reorder Needed'], 
        forecast_date.strftime('%Y-%m-01'), 
        "No Order Needed"
    )

    all_forecasts.append(monthly_forecast)

# Combine both months into one DataFrame
final_forecast = pd.concat(all_forecasts, ignore_index=True)

# **Ensure the predictions directory exists**
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# Save predictions to CSV
output_path = os.path.join(output_dir, "multi_month_demand_forecast.csv")
final_forecast.to_csv(output_path, index=False)

print(f"Multi-month demand forecast saved to {output_path}")
