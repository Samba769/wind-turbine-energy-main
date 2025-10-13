import pandas as pd
import pickle
import numpy as np

# Load model
with open("models/wind_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data
df_test = pd.read_csv("data/processed/wind_data_cleaned.csv")

# Predict
X_test = df_test[['wind_speed', 'temperature', 'humidity']]
predictions = model.predict(X_test)

df_test['predicted_energy'] = predictions
print(df_test.head())
