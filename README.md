# Module-E_project

> **A weekly, SKU-level market trend dataset for agricultural implements, integrating demand, pricing, cost structure, and material economics for forecasting and operational analytics.**

## 📌 Overview
This project applies Machine Learning (Random Forest) and Deep Learning (LSTM) techniques to forecast weekly sales for an agricultural manufacturing business. It is designed to help stakeholders anticipate demand fluctuations, optimize inventory, and analyze market trends using historical sales data.

## 🚀 Key Features
* **Dual Model Support:** Switch seamlessly between **Random Forest Regressor** (for interpretability) and **LSTM Neural Networks** (for sequence modeling).
* **Modular Architecture:** Clean separation of data processing, model definitions, and training logic.
* **Config-Driven:** Hyperparameters and settings are managed via JSON configuration files, allowing for easy experimentation without code changes.
* **Comprehensive Metrics:** Evaluates performance using **R² Score**, **RMSE** (Root Mean Squared Error), and **MAE** (Mean Absolute Error).

## 📂 Project Structure
```text
.
├── configs/
│   └── config.json       # Hyperparameters (epochs, batch size, model type)
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
│   └── weekly_dataset_with_total_units_sold       # Raw sales data
├── tests/                # Unit tests for data and model integrity
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
