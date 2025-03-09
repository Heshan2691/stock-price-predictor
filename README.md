# stock-price-predictor

Stock Price Prediction with LSTM


Date: March 9, 2025

Author: Team Cyferlink


Project Overview
This repository contains a Jupyter Notebook implementing a stock price prediction model using a Long Short-Term Memory (LSTM) neural network. The model forecasts the closing prices of an unspecified stock for the five trading days following December 27, 2024 (i.e., December 30, 2024, to January 3, 2025), based on historical data from March 17, 1980, to December 27, 2024. The dataset, question4-stock-data.csv, includes daily stock metrics such as Close, High, Low, Open, Adj Close, and Volume. The model leverages engineered features like lagged prices, rolling statistics, and volume changes to generate predictions ranging from $181.63 to $200.09.

Repository Contents
stock_price_prediction.ipynb: The main Jupyter Notebook with data preprocessing, feature engineering, model training, and forecasting code.
question4-stock-data.csv: The dataset (11,291 rows) used for training and forecasting.
README.md: This file, providing project details and instructions.
[Optional] requirements.txt: List of Python dependencies (if included).
Dataset Description
Source: question4-stock-data.csv
Time Period: March 17, 1980, to December 27, 2024
Columns:
Date: Trading date
Adj Close: Adjusted closing price
Close: Closing price (target variable)
High: Daily high price
Low: Daily low price
Open: Opening price
Volume: Trading volume
Size: 11,291 rows
Missing Values: Sporadic gaps in all columns, handled by dropping incomplete rows during preprocessing.
Sample Data (Last 5 Days)
Date	Close ($)	Volume
2024-12-20	178.17	425,700
2024-12-23	180.45	422,700
2024-12-24	181.43	168,600
2024-12-26	197.36	1,281,200
2024-12-27	199.52	779,500
Prerequisites
To run the notebook, ensure you have the following installed:

Python: 3.8+
Jupyter Notebook: For interactive execution
Dependencies:
pandas
numpy
scikit-learn
tensorflow
matplotlib (optional, for visualizations)
Install dependencies using pip:

bash

Collapse

Wrap

Copy
pip install pandas numpy scikit-learn tensorflow matplotlib
Or, if a requirements.txt file is provided:

bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Setup Instructions
Clone the Repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
Place the Dataset:
Ensure question4-stock-data.csv is in the same directory as the notebook.
Launch Jupyter Notebook:
bash

Collapse

Wrap

Copy
jupyter notebook
Open stock_price_prediction.ipynb in your browser.
Usage
Run the Notebook:
Execute all cells sequentially to preprocess the data, train the LSTM model, and generate predictions.
The notebook outputs a 5-day forecast for December 30, 2024, to January 3, 2025.
Key Sections:
Data Loading & Preprocessing: Loads the CSV, handles missing values, and engineers features.
Model Training: Trains the LSTM on 80% of the data (1980–~2019).
Forecasting: Predicts closing prices using the last 5 days of 2024 data.
Results: Displays predictions (e.g., $200.09, $186.11, $185.42, $181.63, $185.20).
Modify Parameters (Optional):
Adjust LSTM units, epochs, or batch size in the model definition.
Change the forecast horizon by modifying the loop in the forecasting section.
Model Details
Architecture:
Input: (1, 10) (1 timestep, 10 features)
LSTM Layer: 100 units, ReLU activation
Dropout: 20%
Dense Layer: 1 unit (output)
Features:
Lagged prices (Lag_1 to Lag_5)
Rolling mean and standard deviation (5-day)
Volume change (5-day)
Day of week
Crash indicator
Optimizer: Adam (learning rate = 0.001)
Loss Function: Mean Squared Error (MSE)
Training: 30 epochs, batch size = 64
Results
The model predicts the following closing prices:

5-Day Forecast (Dec 28, 2024 - Jan 1, 2025):
        Date  Predicted_Close
0 2024-12-28       188.494110
1 2024-12-29       178.674942
2 2024-12-30       176.003876
3 2024-12-31       174.920410
4 2025-01-01       174.086365

Limitations
No Validation Metrics: Accuracy unverified due to lack of test set evaluation.
Volume Simplification: Assumes constant volume (779,500) post-December 27.
Holiday Handling: Requires explicit adjustment for non-trading days like January 1, 2025.
Data Gaps: Missing values reduce usable data after preprocessing.
Future Improvements
Add performance metrics (RMSE, MAE, R²) using a 2024 test set.
Incorporate external data (e.g., market indices, sentiment from X posts).
Use a holiday calendar to refine trading day forecasts.
Optimize hyperparameters via grid search.
Model volume dynamically with a secondary model.
