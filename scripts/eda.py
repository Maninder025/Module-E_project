# scripts/eda.py
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# Fix imports to allow importing from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from config to get the correct file path
import json
with open('configs/config.json', 'r') as f:
    config = json.load(f)

FILE_PATH = config['data_config']['file_path']

def analyze_data():
    print(f"--- Analyzing Data from: {FILE_PATH} ---")
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # Load Data
    df = pd.read_excel('src/weekly_dataset_with_total_unit_sold.xlsx', sheet_name='Sheet1')
    
    # 1. Show First Few Rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # 2. Basic Statistics
    print("\nStatistics:")
    print(df.describe())
    
    # 3. Check for Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # 4. Plotting
    # We assume the column is 'Total Unit Sold' based on your dataset
    target_col = 'Total Unit Sold'
    if target_col in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(df['Date']), df[target_col], color='purple', label='Sales')
        plt.title('Raw Sales Data Distribution')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.legend()
        plt.grid(True)
        print("\nDisplaying plot window...")
        plt.show()
    else:
        print(f"\nWarning: Column '{target_col}' not found. Skipping plot.")

if __name__ == "__main__":
    analyze_data()