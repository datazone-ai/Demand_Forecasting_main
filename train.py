from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load feature-engineered data
feature_data = pd.read_csv('feature_data.csv')

# Prepare features and target
X = feature_data.drop(['Demand', 'Date'], axis=1)
X = pd.get_dummies(X, columns=['SKU'], drop_first=True)  # One-hot encode SKU
y = feature_data['Demand']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Save the model (using joblib)
import joblib
joblib.dump(model, 'random_forest_model.pkl')