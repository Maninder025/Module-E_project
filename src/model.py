from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor

# --- Option 1: LSTM Model (Deep Learning) ---
def build_lstm_model(input_shape):
    """
    Expects input_shape to be 3D: (samples, time_steps, features)
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# --- Option 2: Random Forest Model (Machine Learning) ---
def build_rf_model(n_estimators=100, random_state=42):
    """
    Builds a standard Random Forest Regressor.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    return model