from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
import sys

app = Flask(__name__)

# =========================
# Paths setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "wind_model.pkl")
# --- Using your new data path ---
DATA_PATH = os.path.join(BASE_DIR, "wind_data_cleaned.csv") 

# =========================
# Load model safely
# =========================
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Loaded model from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    print("[ERROR] Please run train.py first to create the model.")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# Define the feature names in the exact order the model was trained on
MODEL_FEATURES = [
    'Wind Speed (m/s)', 
    'Theoretical_Power_Curve (KWh)', 
    'Wind Direction (°)',
    'month',
    'day',
    'hour'
]

# --- IMPORTANT: Set your real column names here ---
# (Based on your image)
RAW_DATE_COL = 'Date/Time'
RAW_TARGET_COL = 'LV ActivePower (kW)' # Or whatever the full name is
RAW_WIND_SPEED_COL = 'Wind Speed (m/s)'
RAW_THEO_POWER_COL = 'Theoretical_Power_Curve (KWh)'
RAW_WIND_DIR_COL = 'Wind Direction (Â°)'

TARGET_COLUMN = 'LV ActivePower (kW)' # This is the *new* name we want

# =========================
# Helper Function for Grid Alerts
# =========================
def get_grid_alert(kw):
    """Creates an alert message based on predicted power (in kW)."""
    # Adjust thresholds as needed for your model's output
    HIGH_SUPPLY_THRESHOLD = 3000    # e.g., > 3000 kW
    LOW_SUPPLY_THRESHOLD = 500      # e.g., < 500 kW

    if kw > HIGH_SUPPLY_THRESHOLD:
        return "ALERT: High supply predicted. Prepare to store or curtail."
    elif kw < LOW_SUPPLY_THRESHOLD:
        return "ALERT: Low supply predicted. Prepare to ramp up other sources."
    else:
        return "Normal: Production within expected range."

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    # Load data safely
    if not os.path.exists(DATA_PATH):
        return f"Data file not found at {DATA_PATH}", 500
    
    df = pd.read_csv(DATA_PATH)
    
    # --- FIX: Rename columns to match what the chart expects ---
    try:
        df.rename(columns={
            RAW_TARGET_COL: TARGET_COLUMN,
            RAW_WIND_SPEED_COL: 'Wind Speed (m/s)',
            RAW_THEO_POWER_COL: 'Theoretical_Power_Curve (KWh)',
            RAW_WIND_DIR_COL: 'Wind Direction (°)'
        }, inplace=True)
    except Exception as e:
         return f"Failed to rename columns for visualization. Check raw column names. Error: {e}", 500
    
    # Check for correct columns *after* renaming
    if TARGET_COLUMN not in df.columns:
        return f"Target column '{TARGET_COLUMN}' not in CSV. Please check renames.", 500
        
    # --- Prepare data for charts ---
    wind_energy_data = df[['Wind Speed (m/s)', TARGET_COLUMN]].rename(columns={TARGET_COLUMN: 'energy_output'}).to_dict('records')
    theo_energy_data = df[['Theoretical_Power_Curve (KWh)', TARGET_COLUMN]].rename(columns={TARGET_COLUMN: 'energy_output'}).to_dict('records')
    dir_energy_data = df[['Wind Direction (°)', TARGET_COLUMN]].rename(columns={TARGET_COLUMN: 'energy_output'}).to_dict('records')

    # --- Correlation data ---
    correlations = {
        'Wind Speed': df['Wind Speed (m/s)'].corr(df[TARGET_COLUMN]),
        'Theoretical Power': df['Theoretical_Power_Curve (KWh)'].corr(df[TARGET_COLUMN]),
        'Wind Direction': df['Wind Direction (°)'].corr(df[TARGET_COLUMN])
    }

    return render_template('visualization.html',
                           wind_energy_data=wind_energy_data,
                           theo_energy_data=theo_energy_data,
                           dir_energy_data=dir_energy_data,
                           correlations=correlations)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- Get ALL 7 inputs from the form (Updated) ---
            location = request.form.get('location', 'Inland')
            wind_speed = float(request.form['wind_speed'])
            theoretical_power = float(request.form['theoretical_power'])
            wind_direction = float(request.form['wind_direction'])
            month = float(request.form['month'])
            day = float(request.form['day'])
            hour = float(request.form['hour'])
            
        except (ValueError, TypeError):
            return "Invalid input. Please enter valid numbers for all fields.", 400

        # Adjust prediction based on location
        location_multiplier = {
            'Coastal': 1.2,
            'Inland': 1.0,
            'Mountain': 1.3
        }.get(location, 1.0)

        # --- Prepare input for model (Updated) ---
        input_data = np.array([[
            wind_speed,
            theoretical_power,
            wind_direction,
            month,
            day,
            hour
        ]])

        try:
            # === SCENARIO 1: MAKE PREDICTION ===
            prediction = model.predict(input_data)[0]
            
            # Apply multiplier and ensure prediction isn't negative
            final_prediction = max(0, prediction * location_multiplier) 
            
            # === SCENARIO 3: GRID ALERT (Single Point) ===
            grid_alert_message = get_grid_alert(final_prediction)

            # --- NEW: Check if sound should play ---
            play_sound = False
            if "ALERT: Low supply" in grid_alert_message:
                play_sound = True
            # -------------------------------------

        except Exception as e:
            return f"Error making prediction: {e}", 500

        return render_template('result.html', 
                               prediction=final_prediction, 
                               location=location,
                               alert=grid_alert_message,
                               play_sound=play_sound) # <-- Pass the new variable here

    # If GET request, just show the prediction form
    return render_template('index.html')


# ==========================================================
# NEW ROUTE FOR SCENARIOS 2 & 3 (TIME SERIES)
# ==========================================================
@app.route('/scenario_dashboard')
def scenario_dashboard():
    try:
        # Load the *entire* dataset
        df = pd.read_csv(DATA_PATH)

        # --- NEW: RENAME COLUMNS TO MATCH MODEL ---
        # (Based on your image)
        try:
            df.rename(columns={
                RAW_WIND_SPEED_COL: 'Wind Speed (m/s)',
                RAW_THEO_POWER_COL: 'Theoretical_Power_Curve (KWh)',
                RAW_WIND_DIR_COL: 'Wind Direction (°)'
            }, inplace=True)
        except Exception as e:
            return f"Failed to rename columns for dashboard. Check raw column names. Error: {e}", 500
        
        # --- NEW: PROCESS THE DATE/TIME COLUMN ---
        # (Based on your image format "01 01 2018 00:00")
        try:
            df[RAW_DATE_COL] = pd.to_datetime(df[RAW_DATE_COL], format='%d %m %Y %H:%M')
        except Exception as e:
            return f"Failed to read Date/Time column. Error: {e}. Check format.", 500

        # --- NEW: CREATE THE MISSING COLUMNS ---
        df['month'] = df[RAW_DATE_COL].dt.month
        df['day'] = df[RAW_DATE_COL].dt.day
        df['hour'] = df[RAW_DATE_COL].dt.hour
        # ------------------------------------------
        
        # --- Simulate a 48-hour "forecast" ---
        forecast_slice = df.head(48).copy()

        # Check if we have enough data
        if len(forecast_slice) < 4: # Need at least 4 hours
             return f"Not enough data in {DATA_PATH} (need at least 4 rows).", 500
            
        # Check for necessary feature columns
        for col in MODEL_FEATURES:
            if col not in forecast_slice.columns:
                return f"Missing required column for prediction: {col}", 500

        # === SCENARIO 1: Generate predictions for all 48 hours ===
        X_forecast = forecast_slice[MODEL_FEATURES]
        predicted_power = model.predict(X_forecast)

        # Create a new DataFrame to hold results
        results_df = forecast_slice[['month', 'day', 'hour']].copy()
        results_df['Predicted_kW'] = np.maximum(0, predicted_power) 
        results_df['id'] = results_df.index 

        # === SCENARIO 2: MAINTENANCE PLANNING ===
        MAINTENANCE_DURATION = 4 # 4-hour window
        rolling_avg = results_df['Predicted_kW'].rolling(window=MAINTENANCE_DURATION).mean()
        best_window_end_index = rolling_avg.idxmin()
        best_window_start_index = best_window_end_index - MAINTENANCE_DURATION + 1
        min_avg_power = rolling_avg.min()
        start_time_info = results_df.loc[best_window_start_index]
        end_time_info = results_df.loc[best_window_end_index]
        
        maintenance_info = {
            'start': f"Day {int(start_time_info['day'])}, Month {int(start_time_info['month'])}, Hour {int(start_time_info['hour'])}",
            'end': f"Day {int(end_time_info['day'])}, Month {int(end_time_info['month'])}, Hour {int(end_time_info['hour'])}",
            'avg_power_kw': min_avg_power
        }

        # === SCENARIO 3: GRID INTEGRATION ALERTS ===
        results_df['Alert'] = results_df['Predicted_kW'].apply(get_grid_alert)
        grid_alerts_df = results_df[results_df['Alert'].str.contains("ALERT")]
        grid_alerts = grid_alerts_df.to_dict('records')
        chart_data = results_df.to_dict('records')

    except Exception as e:
        return f"Error processing scenario dashboard: {e}", 500

    return render_template('scenario_dashboard.html',
                           maintenance=maintenance_info,
                           alerts=grid_alerts,
                           chart_data=chart_data)

# =========================
# Run app
# =========================
if __name__ == '__main__':
    app.run(debug=True)
