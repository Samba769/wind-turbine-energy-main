import pandas as pd
import pickle
import numpy as np

# Load model
try:
    with open("models/wind_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: model not found. Run train.py first.")
    exit()


# Load test data
try:
    df_test = pd.read_csv("T1.csv")
except FileNotFoundError:
    print("ERROR: T1.csv not found.")
    exit()

# --- Must perform the same feature engineering as in train.py ---
try:
    df_test['Date/Time'] = pd.to_datetime(df_test['Date/Time'], format='%d %m %Y %H:%M')
    df_test['month'] = df_test['Date/Time'].dt.month
    df_test['day'] = df_test['Date/Time'].dt.day
    df_test['hour'] = df_test['Date/Time'].dt.hour
    df_test.dropna(inplace=True)
except Exception as e:
    print(f"ERROR: Could not process Date/Time column. {e}")
    exit()


# --- FIXED: Features must match the 6 features used in training ---
FEATURE_COLUMNS = [
    'Wind Speed (m/s)', 
    'Theoretical_Power_Curve (KWh)', 
    'Wind Direction (°)',
    'month',
    'day',
    'hour'
]
X_test = df_test[FEATURE_COLUMNS] 

# Predict
predictions = model.predict(X_test)

df_test['predicted_power_kW'] = predictions

# Print the first few rows with predictions
print("--- Predictions complete ---")
print(df_test[['Date/Time', 'LV ActivePower (kW)', 'predicted_power_kW']].head())
