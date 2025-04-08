import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(input_path, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Convert date columns to datetime format
    date_columns = ['Manufacturing Date', 'Expiration Date', 'Order Date', 'Delivery Date', 'Shipping Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Encode SKU column
    le = LabelEncoder()
    df['SKU'] = le.fit_transform(df['SKU'])
    
    # Convert Season column to numerical values
    season_mapping = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    df['Season'] = df['Season'].map(season_mapping).fillna(-1).astype(int)
    
    # Feature Engineering - Order Date based features
    df['Order_Day'] = df['Order Date'].dt.day
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Weekday'] = df['Order Date'].dt.weekday  # Monday=0, Sunday=6

    # Calculate Days Since Last Order
    df['Days_Since_Last_Order'] = (df['Order Date'] - df['Order Date'].shift(1)).dt.days.fillna(0)
    
    # Feature Engineering - Expiry & Lead Time
    df['Days_Until_Expiry'] = (df['Expiration Date'] - pd.to_datetime('today')).dt.days
    df['Lead_Time'] = (df['Delivery Date'] - df['Order Date']).dt.days

    # Adjust stock values before selecting columns
    df['Stock In'] = np.where(df['Stock Balance'] < 0, abs(df['Stock Balance']), df['Stock In'])
    df['Stock Balance'] = df['Stock In'] - df['Stock Out']
    df['Stock Balance'] = df['Stock Balance'].clip(lower=0)

    # Select relevant columns
    final_columns = [
        'SKU', 'Stock In', 'Stock Out', 'Stock Balance', 'Days_Until_Expiry', 'Lead_Time',
        'Season', 'Holiday', 'Promotion', 'Lag_1', 'Lag_2', 'Lag_3', 
        'Moving Average', 'Demand Forecast', 'Order_Day', 'Order_Month', 'Order_Weekday', 'Days_Since_Last_Order'
    ]
    
    # Ensure all required columns exist before filtering
    df = df[[col for col in final_columns if col in df.columns]]
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Data saved to {output_path}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("original_data/final_medicine_inventory_sales_data.csv", "processed_data/medicine_inventory_cleaned.csv")
