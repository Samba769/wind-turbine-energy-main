import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Set the path to your data file here
file_path = "wind_data_cleaned.csv" 
# Set the name of the column you want to predict
target_column = 'LV ActivePower (kW)'
# ---------------------

print(f"--- Starting Data Check for: {file_path} ---")

try:
    # Use the file_path variable to load the data
    df = pd.read_csv(file_path) 
except FileNotFoundError:
    # This error message is now accurate
    print(f"\nERROR: File not found at {file_path}") 
    print("Please check the path and run again.")
    exit()
except Exception as e:
    print(f"\nAn unexpected error occurred while loading the file: {e}")
    exit()

print("\n" + "="*60 + "\n")
print("--- 1. Data Types and Missing Values ---")
print(df.info())

print("\n" + "="*60 + "\n")
print("--- 2. Data Statistics (Look for strange values) ---")
print(df.describe())

print("\n" + "="*60 + "\n")
print(f"--- 3. Correlation with Target Column: '{target_column}' ---")

try:
    # Check if the target column exists first
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in the CSV.")

    correlations = df.corr(numeric_only=True)[target_column].sort_values()
    print(correlations)
    
except KeyError as e:
    print(f"ERROR: {e}")
    print("Available columns are:", list(df.columns))
except TypeError:
    print(f"ERROR: Could not calculate correlation for '{target_column}'.")
    print("Check if all data columns are numeric.")
except Exception as e:
    print(f"An unexpected error occurred during correlation: {e}")

print("\n" + "="*60 + "\n")
print("--- Data Check Complete ---")
