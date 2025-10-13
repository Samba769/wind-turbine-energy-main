from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# Load the trained model
with open("models/wind_model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page (default)
@app.route('/')
def home():
    return render_template('home.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Visualization page
@app.route('/visualization')
def visualization():
    # Load data for visualization
    df = pd.read_csv("data/processed/wind_data_cleaned.csv")

    # Prepare data for charts
    wind_energy_data = df[['wind_speed', 'energy_output']].to_dict('records')
    temp_energy_data = df[['temperature', 'energy_output']].to_dict('records')
    humidity_energy_data = df[['humidity', 'energy_output']].to_dict('records')

    # Correlation data
    correlations = {
        'wind_speed': df['wind_speed'].corr(df['energy_output']),
        'temperature': df['temperature'].corr(df['energy_output']),
        'humidity': df['humidity'].corr(df['energy_output']),
        'pressure': df['pressure'].corr(df['energy_output'])
    }

    return render_template('visualization.html',
                         wind_energy_data=wind_energy_data,
                         temp_energy_data=temp_energy_data,
                         humidity_energy_data=humidity_energy_data,
                         correlations=correlations)

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        location = request.form['location']
        wind_speed = float(request.form['wind_speed'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])

        # Adjust prediction based on location
        location_multiplier = {
            'Coastal': 1.2,  # Higher wind speeds
            'Inland': 1.0,   # Standard
            'Mountain': 1.3  # Higher wind speeds in mountains
        }.get(location, 1.0)

        # Prepare input for model
        input_data = np.array([[temperature, humidity, wind_speed, pressure]])

        # Make prediction
        prediction = model.predict(input_data)[0] * location_multiplier

        return render_template('result.html', prediction=prediction, location=location)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
