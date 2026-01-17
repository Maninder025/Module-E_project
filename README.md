# Module-E_project

> **A weekly, SKU-level market trend dataset for agricultural implements, integrating demand, pricing, cost structure, and material economics for forecasting and operational analytics.**

## 📌 Overview
This project applies Machine Learning (Random Forest) and Deep Learning (LSTM) techniques to forecast weekly sales for an agricultural manufacturing business. It is designed to help stakeholders anticipate demand fluctuations, optimize inventory, and analyze market trends using historical sales data.

## 🚀 Key Features
* **Dual Model Support:** Switch seamlessly between **Random Forest Regressor** (for interpretability) and **LSTM Neural Networks** (for sequence modeling).
* **Modular Architecture:** Clean separation of data processing, model definitions, and training logic.
* **Config-Driven:** Hyperparameters and settings are managed via JSON configuration files, allowing for easy experimentation without code changes.
* **Comprehensive Metrics:** Evaluates performance using **R² Score**, **RMSE** (Root Mean Squared Error), and **MAE** (Mean Absolute Error).


## 🛠️ Installation
Clone the repository:

git clone [https://github.com/yourusername/agricultural-forecasting.git](https://github.com/Maninder025/Module-E_project)
cd Module-E_project
## Install dependencies:

pip install -r requirements.txt


## ⚙️ Usage
### 1. Configure the Experiment
Open configs/config.json to select your model and settings:

### JSON

{
    "model_config": {
        "model_type": "RF",   // Options: "RF" or "LSTM"
        "rf_estimators": 100,
        "lstm_units": 50
    }
}
### 2. Train the Model
Run the main training script. This will train the model, evaluate it, and save the artifacts to the models/ folder.

python src/train.py
### 3. Visual Analysis (EDA)
To view sales trends and data distribution before training:

python scripts/eda.py

### 4. Make Predictions (Inference)
To predict sales on a new dataset (e.g., 2025 data):

python scripts/predict.py


##📊 Model Performance
The models are evaluated on the following metrics:

### R² Score:
Measures how well the model explains the variance in the data (Closer to 1.0 is better).

### MAE:
The average magnitude of errors in unit sales.

## 🧪 Testing
To ensure the pipeline is functioning correctly, run the unit tests:

python -m unittest discover tests

## 📂 Project Structure
```text
.
├── configs/
│   └── config.json       # Hyperparameters (epochs, batch size, model type)
├── data/
│   └── dataset.csv       # Raw sales data
├── models/
│   ├── rf_model.pkl      # Saved Random Forest model
│   └── scaler.pkl        # Saved Data Scaler (critical for inference)
├── scripts/
│   ├── eda.py            # Exploratory Data Analysis & Plotting
│   └── predict.py        # Inference script for new data
├── src/
│   ├── data.py           # Data loading, preprocessing, and windowing
│   ├── model.py          # Architecture definitions (LSTM & RF)
│   └── train.py          # Main training loop
├── tests/                # Unit tests for data and model integrity
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
