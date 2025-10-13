import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/processed/wind_data_cleaned.csv")

# Features and target
X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = df['energy_output']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("models/wind_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
