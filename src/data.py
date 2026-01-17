import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def load_and_preprocess_data(filepath, look_back=3):
    # 1. Load Data
    data = pd.read_csv(filepath)
    
    # 2. Date Processing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # 3. Resample to Weekly (Summing 'Total Unit Sold')
    # Note: Using the column name from your script
    weekly_sales = data['Total Unit Sold'].resample('W').sum()
    
    # 4. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    weekly_sales_scaled = scaler.fit_transform(weekly_sales.values.reshape(-1, 1))
    
    # 5. Create Sequences
    X, y = create_sequences(weekly_sales_scaled, look_back)
    
    # 6. Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    # 7. Split Data (80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    
    return X_train, y_train, X_test, y_test, scaler