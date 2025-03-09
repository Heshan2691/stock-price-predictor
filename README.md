# stock-price-predictor

Stock Price Prediction with LSTM
================================
Date: March 9, 2025
Author: Team Cyferlink

Project Overview
----------------
This document accompanies a Jupyter Notebook implementing a stock price prediction model using a Long Short-Term Memory (LSTM) neural network. The model forecasts closing prices for an unspecified stock over five trading days following December 27, 2024 (December 30, 2024, to January 3, 2025). It uses historical data from March 17, 1980, to December 27, 2024, stored in `question4-stock-data.csv`. Predictions range from $181.63 to $200.09, based on features like lagged prices, rolling statistics, and volume changes.

Repository Contents
-------------------
- `stock_price_prediction.ipynb`: Main notebook with preprocessing, training, and forecasting code.
- `question4-stock-data.csv`: Dataset (11,291 rows) for training and prediction.
- `README.doc`: This file, providing project details and instructions.
- [Optional] `requirements.txt`: List of Python dependencies.

Dataset Description
-------------------
Source: `question4-stock-data.csv`
Time Period: March 17, 1980, to December 27, 2024
Columns:
  - Date: Trading date
  - Adj Close: Adjusted closing price
  - Close: Closing price (target variable)
  - High: Daily high price
  - Low: Daily low price
  - Open: Opening price
  - Volume: Trading volume
Size: 11,291 rows
Missing Values: Sporadic gaps in all columns, handled by dropping incomplete rows.

Sample Data (Last 5 Days):
  Date         | Close ($) | Volume
  ------------ | ----------|---------
  2024-12-20   | 178.17    | 425,700
  2024-12-23   | 180.45    | 422,700
  2024-12-24   | 181.43    | 168,600
  2024-12-26   | 197.36    | 1,281,200
  2024-12-27   | 199.52    | 779,500

Prerequisites
-------------
To run the notebook, install:
- Python: 3.8+
- Jupyter Notebook: For interactive execution
- Dependencies:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - matplotlib (optional, for visualizations)

Install via pip:
pip install pandas numpy scikit-learn tensorflow matplotlib

Or, with `requirements.txt`:
pip install -r requirements.txt

Setup Instructions
------------------
1. Clone the Repository:
   git clone https://github.com/Heshan2691/stock-price-predictor.git
   cd stock-price-predictor

2. Place the Dataset:
   Ensure `question4-stock-data.csv` is in the notebook directory.

3. Launch Jupyter Notebook:
   jupyter notebook
   Open `stock_price_prediction.ipynb`.

Usage
-----
1. Run the Notebook:
   - Execute cells sequentially to preprocess data, train the LSTM, and predict prices.
   - Outputs a 5-day forecast for December 30, 2024, to January 3, 2025.

2. Key Sections:
   - Data Loading & Preprocessing: Loads CSV, handles missing values, engineers features.
   - Model Training: Trains LSTM on 80% of data (1980–~2019).
   - Forecasting: Predicts using last 5 days of 2024.
   - Results: Displays predictions (e.g., $200.09, $186.11, $185.42, $181.63, $185.20).

3. Modify Parameters (Optional):
   - Adjust LSTM units, epochs, or batch size.
   - Change forecast horizon in the forecasting loop.

Model Details
-------------
Architecture:
  - Input: (1, 10) (1 timestep, 10 features)
  - LSTM Layer: 100 units, ReLU activation
  - Dropout: 20%
  - Dense Layer: 1 unit (output)
Features:
  - Lagged prices (Lag_1 to Lag_5)
  - Rolling mean and standard deviation (5-day)
  - Volume change (5-day)
  - Day of week
  - Crash indicator
Optimizer: Adam (learning rate = 0.001)
Loss Function: Mean Squared Error (MSE)
Training: 30 epochs, batch size = 64

Results
-------
5-Day Forecast (Dec 28, 2024 - Jan 1, 2025):
Date       | Predicted Close ($)
  -----------|-------------------
  2024-12-28 | 188.494110
  2024-12-29 | 178.674942
  2024-01-30 | 176.003876
  2024-01-31 | 174.920410
  2025-01-01 | 174.086365

Limitations
-----------
- No Validation Metrics: Accuracy unverified without test set evaluation.
- Volume Simplification: Assumes constant volume (779,500) post-December 27.
- Holiday Handling: Requires adjustment for non-trading days like January 1, 2025.
- Data Gaps: Missing values reduce usable data.

Future Improvements
-------------------
- Add performance metrics (RMSE, MAE, R²) using 2024 test set.
- Incorporate external data (e.g., market indices, X post sentiment).
- Use holiday calendar for accurate trading days.
- Optimize hyperparameters via grid search.
- Model volume dynamically.

