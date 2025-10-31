import pandas as pd
import pickle
import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# --- Import the models you want to compare ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Suppress warnings
warnings.filterwarnings('ignore')

print("Starting model training process...")
os.makedirs("models", exist_ok=True)

# 1️⃣ Load Data
try:
    data = pd.read_csv('wind_data_cleaned.csv') 
except FileNotFoundError:
    print("ERROR: 'wind_data_cleaned.csv' not found.")
    exit()

# 2️⃣ Feature Engineering
print("--- 1. Starting Feature Engineering ---")
try:
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%d %m %Y %H:%M')
    data['month'] = data['Date/Time'].dt.month
    data['day'] = data['Date/Time'].dt.day
    data['hour'] = data['Date/Time'].dt.hour
    data.dropna(inplace=True)
except KeyError:
    print("ERROR: 'Date/Time' column not found.")
    exit()
except ValueError:
    print("ERROR: 'Date/Time' column not in format 'dd mm yyyy HH:MM'")
    exit()
print("--- Feature Engineering Complete ---")

# 3️⃣ Define Features and Target
FEATURE_COLUMNS = [
    'Wind Speed (m/s)', 
    'Theoretical_Power_Curve (KWh)', 
    'Wind Direction (°)',
    'month', 'day', 'hour'
]
TARGET_COLUMN = 'LV ActivePower (kW)'

# Check for missing columns
missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
if missing_cols:
    print(f"ERROR: Missing columns in CSV: {missing_cols}")
    exit()

X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

# 4️⃣ Main Train-Test Split (Lock away the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- Data split: {len(X_train)} training samples, {len(X_test)} test samples ---")

# 5️⃣ Initialize Models
models = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# 6️⃣ Compare models using Cross-Validation *on the training set*
print("\n--- 2. Comparing models using 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
best_model_name = ""
best_cv_r2 = -float('inf')

for name, model in models.items():
    print(f"Running CV for: {name}...")
    # Perform CV on the training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    
    print(f"  Mean R2: {mean_r2:.4f} (Std: +/- {std_r2:.4f})")
    
    cv_results.append({
        "Model": name,
        "Mean CV R2": mean_r2,
        "Std Dev": std_r2
    })
    
    # Check if this is the best model
    if mean_r2 > best_cv_r2:
        best_cv_r2 = mean_r2
        best_model_name = name

# Print CV summary
print("\n--- 3. Cross-Validation Summary ---")
cv_results_df = pd.DataFrame(cv_results).sort_values(by="Mean CV R2", ascending=False)
print(cv_results_df)

print(f"\n--- Best model based on CV: {best_model_name} ---")

# 7️⃣ Train the *best* model on the *entire* training set
print(f"\n--- 4. Training final '{best_model_name}' model... ---")
final_model = models[best_model_name]
final_model.fit(X_train, y_train)
print("--- Final Model Training Complete ---")

# 8️⃣ Evaluate the final model on the *locked-away* test set
print("\n--- 5. Final Evaluation on Test Set ---")
y_pred = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  Final Test R-squared (R2): {final_r2:.4f}")
print(f"  Final Test RMSE: {final_rmse:.2f}")
print("---------------------------------")

# 9️⃣ Save the final, trained model
model_path = "models/wind_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

print(f"Final model saved successfully to {model_path}!")
print("Training process completed.")
