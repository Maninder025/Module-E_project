import unittest
import numpy as np
import sys
import os

# Fix path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_and_preprocess_data

class TestDataPipeline(unittest.TestCase):
    
    def test_data_shapes(self):
        """
        Test if data is loaded and split into correct shapes
        """
        # Use your actual file path or a dummy one if you have a test file
        file_path = 'weekly_dataset_with_total_unit_sold.xlsx - Sheet1.csv'
        look_back = 3
        
        # Only run this if file exists, otherwise skip
        if os.path.exists(file_path):
            X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(file_path, look_back)
            
            # Check if we got data back
            self.assertGreater(len(X_train), 0, "X_train should not be empty")
            self.assertGreater(len(y_train), 0, "y_train should not be empty")
            
            # Check 3D shape for LSTM (Samples, TimeSteps, Features)
            # We expect X_train to be 3 dimensions
            self.assertEqual(len(X_train.shape), 3, "X_train should be 3D (Samples, TimeSteps, Features)")
            
            # Check if values are scaled between 0 and 1
            self.assertTrue(np.max(X_train) <= 1.0001, "Data should be scaled max <= 1")
            self.assertTrue(np.min(X_train) >= -0.0001, "Data should be scaled min >= 0")
        else:
            print("Skipping data test: Dataset file not found.")

if __name__ == '__main__':
    unittest.main()