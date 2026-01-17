import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# 1. Setup path to import from src if needed (and to find config)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_config():
    """Load configuration from the JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def create_sequences(data, look_back):
    """
    Create sequences for LSTM/RF input.
    Matches the logic used in training.
    """
    X = []
    # We only create X (input) because we might not have Y (target) for the future
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
    return np.array(X)

def main():
    # --- 1. Setup & Config ---
    config = load_config()
    
    # You can change this file path to whatever new data file you want to predict on
    NEW_DATA_PATH = 'src/weekly_dataset_with_total_unit_sold.xlsx - Sheet1.csv' 
    
    MODEL_TYPE = config['model_config']['model_type']
    LOOK_BACK = config['data_config']['look_back']
    
    print(f"--- Starting Prediction using {MODEL_TYPE} Model ---")

    # --- 2. Load Scaler & Model ---
    # Define paths
    base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    rf_path = os.path.join(base_path, 'rf_model.pkl')
    lstm_path = os.path.join(base_path, 'lstm_model.h5')

    # Load Scaler
    if not os.path.exists(scaler_path):
        print("CRITICAL ERROR: 'scaler.pkl' not found in models/ folder.")
        print("Please run 'src/train.py' first to generate the scaler.")
        return
    scaler = joblib.load(scaler_path)

    # Load Model
    model = None
    if MODEL_TYPE == 'RF':
        if os.path.exists(rf_path):
            model = joblib.load(rf_path)
        else:
            print(f"Error: Random Forest model not found at {rf_path}")
            return
    elif MODEL_TYPE == 'LSTM':
        if os.path.exists(lstm_path):
            model = load_model(lstm_path)
        else:
            print(f"Error: LSTM model not found at {lstm_path}")
            return

    # --- 3. Process New Data ---
    if not os.path.exists(NEW_DATA_PATH):
        print(f"Error: Data file '{NEW_DATA_PATH}' not found.")
        return

    print("Processing data...")
    df = pd.read_csv(NEW_DATA_PATH)
    
    # Ensure Date parsing matches your training logic
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Aggregating to weekly level
    weekly_sales = df['Total Unit Sold'].resample('W').sum()
    
    # Scale the data using the TRAINED scaler (do not fit new scaler)
    # This ensures 100 units today means the same as 100 units during training
    scaled_data = scaler.transform(weekly_sales.values.reshape(-1, 1))

    # Create window sequences
    # Note: If you want to predict the very NEXT week after your data ends,
    # you take the last 'look_back' weeks.
    
    # Here we predict on the existing historical data to see how it performs
    X_new = create_sequences(scaled_data, LOOK_BACK)

    # Handle Shape differences
    if MODEL_TYPE == 'LSTM':
        # LSTM needs 3D: (Samples, TimeSteps, Features)
        X_new = np.reshape(X_new, (X_new.shape[0], 1, X_new.shape[1]))
    elif MODEL_TYPE == 'RF':
        # RF needs 2D: (Samples, Features)
        nsamples, nx = X_new.shape
        X_new = X_new.reshape((nsamples, nx))

    # --- 4. Predict ---
    print("Making predictions...")
    predictions_scaled = model.predict(X_new)

    # Reshape if necessary for inverse transform
    if MODEL_TYPE == 'RF':
        predictions_scaled = predictions_scaled.reshape(-1, 1)

    # Convert back to real units
    predictions_actual = scaler.inverse_transform(predictions_scaled)

    # --- 5. Output Results ---
    print("\nSample Predictions (First 5 weeks available):")
    for i in range(min(5, len(predictions_actual))):
        print(f"Week {i+1}: {predictions_actual[i][0]:.2f} Units")

    # Optional: Save to CSV
    output_df = pd.DataFrame(predictions_actual, columns=['Predicted_Sales'])
    output_df.to_csv('predictions.csv', index=False)
    print("\nFull predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    main()
