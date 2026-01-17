import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error, r2_score
import json

# Import your custom modules
from src.data import load_and_preprocess_data
from src.model import build_lstm_model, build_rf_model

def main():
    
    # ==========================================
    # --- CONFIGURATION SECTION ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Use variables from config instead of hardcoding
    FILE_PATH = config['data_config']['file_path']
    LOOK_BACK = config['data_config']['look_back']
    MODEL_TYPE = config['model_config']['model_type']
    EPOCHS = config['train_config']['epochs']
    BATCH_SIZE = config['train_config']['batch_size']
    # ==========================================
    
    # --- 1. Get Data ---
    print("Loading and processing data...")
    # X_train comes in as 3D shape: (Samples, TimeSteps, Features)
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(FILE_PATH, LOOK_BACK)
    
    # --- 2. Train Model (Logic splits based on MODEL_TYPE) ---
    model = None
    
    if MODEL_TYPE == "LSTM":
        print(f"Training {MODEL_TYPE} model...")
        model = build_lstm_model(X_train.shape)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        # Predictions (LSTM accepts 3D data natively)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
    elif MODEL_TYPE == "RF":
        print(f"Training {MODEL_TYPE} model...")
        model = build_rf_model()
        
        # RESHAPING FOR RANDOM FOREST
        # Random Forest cannot handle 3D data (Samples, TimeSteps, Features).
        # We flatten it to 2D (Samples, TimeSteps * Features).
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples, nx*ny))
        
        nsamples_test, nx_test, ny_test = X_test.shape
        X_test_2d = X_test.reshape((nsamples_test, nx_test*ny_test))
        
        # Train
        model.fit(X_train_2d, y_train)
        
        # Predict
        train_predict = model.predict(X_train_2d)
        test_predict = model.predict(X_test_2d)
        
        # RESHAPE OUTPUT
        # RF returns shape (N,), but our scaler expects (N, 1)
        train_predict = train_predict.reshape(-1, 1)
        test_predict = test_predict.reshape(-1, 1)

    # --- 3. Inverse Transform (Un-scale) ---
    # Convert predictions back to original units (e.g., actual sales numbers)
    train_predict = scaler.inverse_transform(train_predict)
    y_train_actual = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform([y_test])
    
    # --- 4. Evaluation Metrics ---
    # Calculate R2 Score for train and test
    train_r2 = r2_score(y_train_actual[0], train_predict[:,0])
    test_r2 = r2_score(y_test_actual[0], test_predict[:,0])
    
    # Calculate MAE
    mae = mean_absolute_error(y_test_actual[0], test_predict[:,0])
    
    print(f"\n--- {MODEL_TYPE} RESULTS ---")
    print(f"Train R2 Score: {train_r2:.2f}")
    print(f"Test R2 Score: {test_r2:.2f}")
    print(f"Test MAE: {mae:.2f}")
    
    # --- 5. Plotting ---
    # Prepare data for plotting
    # We create empty arrays filled with NaN so the plots align correctly on the x-axis
    
    # Structure: [Original Data]
    original_data = scaler.inverse_transform(np.concatenate((y_train, y_test)).reshape(-1,1))
    
    # Structure: [nan, nan, ... Train Predictions ..., nan, nan]
    train_predict_plot = np.empty_like(original_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[LOOK_BACK:len(train_predict)+LOOK_BACK, :] = train_predict
    
    # Structure: [nan, nan, ... nan, ... Test Predictions ...]
    test_predict_plot = np.empty_like(original_data)
    test_predict_plot[:, :] = np.nan
    
    # Calculate start index for test predictions
    # (Train len + Lookback for first sequence + 1 to shift)
    test_start_idx = len(train_predict) + (LOOK_BACK * 2) + 1
    
    # Fill from the end to ensure alignment
    test_predict_plot[len(original_data) - len(test_predict):, :] = test_predict

    plt.figure(figsize=(12,6))
    plt.title(f"Sales Forecasting ({MODEL_TYPE})")
    plt.plot(original_data, label='Actual Sales', color='blue')
    plt.plot(train_predict_plot, label='Training Prediction', color='orange')
    plt.plot(test_predict_plot, label='Test Prediction', color='green')
    plt.xlabel('Weeks')
    plt.ylabel('Total Units Sold')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()