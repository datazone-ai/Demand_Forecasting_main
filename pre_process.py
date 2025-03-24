import pandas as pd

df = pd.read_csv('./data/synthetic_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
def create_features(df):
    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Lag features (demand from previous months)
    df['Lag1'] = df.groupby('SKU')['Demand'].shift(1)
    
    # Rolling statistics (3-month average demand)
    df['RollingMean3'] = df.groupby('SKU')['Demand'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # Drop rows with missing values (due to lag/rolling features)
    df = df.dropna()
    
    return df

# Apply feature engineering
feature_data = create_features(df)

# Save feature-engineered data
feature_data.to_csv('feature_data.csv', index=False)

print("Feature-Engineered Data Sample:")
print(feature_data.head())