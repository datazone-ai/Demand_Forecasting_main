import pandas as pd
import numpy as np

# Parameters
num_skus = 150
num_months = 24
start_date = '2021-01-01'

# Generate SKUs
skus = [f'SKU_{i+1}' for i in range(num_skus)]

# Generate dates
dates = pd.date_range(start=start_date, periods=num_months, freq='M')

# Generate synthetic data
data = []
for sku in skus:
    base_demand = np.random.randint(100, 500)  # Base demand for each SKU
    for date in dates:
        # Simulate seasonality (e.g., higher demand in December)
        seasonality = 1.5 if date.month == 12 else 1.0
        # Simulate random noise
        noise = np.random.normal(1, 0.1)
        # Simulate promotions (randomly)
        promotion = 1.5 if np.random.rand() > 0.8 else 1.0
        # Simulate holidays (randomly)
        holiday = 1 if np.random.rand() > 0.9 else 0
        # Calculate final demand
        demand = int(base_demand * seasonality * promotion * noise)
        data.append([sku, date, demand, int(promotion > 1), holiday])

# Create DataFrame
df = pd.DataFrame(data, columns=['SKU', 'Date', 'Demand', 'Promotion', 'Holiday'])

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)

print("Synthetic Data Sample:")
print (df.head())


df.shape

