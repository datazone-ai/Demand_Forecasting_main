import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import joblib

from train import X_train

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Function to generate forecasts
def generate_forecasts(model, feature_data, future_months=3):
    forecasts = []
    last_date = feature_data['Date'].max()
    skus = feature_data['SKU'].unique()

    for sku in skus:
        sku_data = feature_data[feature_data['SKU'] == sku].copy()
        for i in range(1, future_months + 1):
            future_date = last_date + timedelta(days=30 * i)
            # Create features for the future date
            future_row = {
                'SKU': sku,
                'Date': future_date,
                'Year': future_date.year,
                'Month': future_date.month,
                'WeekOfYear': future_date.isocalendar().week,
                'Lag1': sku_data['Demand'].iloc[-1],  # Use last known demand as lag
                'RollingMean3': sku_data['Demand'].tail(3).mean(),  # Use rolling mean
                'Promotion': 0,  # Assume no promotion by default
                'Holiday': 0  # Assume no holiday by default
            }
            sku_data = pd.concat([sku_data, pd.DataFrame([future_row])], ignore_index=True)

            # Prepare features for prediction
            X_future = pd.DataFrame([future_row])
            # Ensure all feature columns are present
            X_future = X_future[X_train.columns]  # Use columns from training data
            # Predict demand
            future_demand = model.predict(X_future)[0]
            forecasts.append([sku, future_date, future_demand])

    return pd.DataFrame(forecasts, columns=['SKU', 'Date', 'Forecasted_Demand'])

# Streamlit App
st.title('Demand Forecasting for DAWAH GROUP')

# Load feature-engineered data
feature_data = pd.read_csv('feature_data.csv')
feature_data['Date'] = pd.to_datetime(feature_data['Date'])

# Generate forecasts
forecasts = generate_forecasts(model, feature_data, future_months=3)

# Display forecasts in a table
st.write("### Forecasted Demand for Next 3 Months")
st.table(forecasts)

# Simulate ERP Integration
st.write("### Simulate ERP Integration")
if st.button('Create Purchase Orders'):
    st.write("Purchase Orders Created Successfully!")
    st.write(forecasts)