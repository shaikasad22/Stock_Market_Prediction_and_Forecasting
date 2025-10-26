Stock Market Prediction and Forecasting Using Stacked LSTM
Project Description
This Jupyter Notebook provides a comprehensive workflow for time series prediction of Apple Inc. (AAPL) stock prices using a deep learning model based on stacked LSTM layers. The notebook demonstrates every step required for data collection, preprocessing, feature engineering, neural network modeling, and visualization, using current best practices in Python and Keras/Tensorflow.​
Key Steps and Workflow
1. Data Collection
Imports historical stock data for AAPL using pandas_datareader and Tiingo API.
Saves data to CSV and loads it as a pandas DataFrame for analysis.
Data columns included: symbol, date, close, high, low, open, volume, and adjusted prices, among others.​
2. Data Preprocessing and Preparation
Extracts 'close' prices for modeling.
Applies MinMaxScaler for normalization. LSTM performance benefits from scaled input.
Splits the time series into training and test portions (approx. 65% training, 35% test).
Converts sequences into supervised learning format using sliding windows (100 timesteps).​
3. Feature Engineering
Reshapes time series into LSTM-suitable 3D format: [samples, timesteps, features].
Defines dataset matrices for network training and testing.​
4. Model Construction
Builds a stacked LSTM using three sequential LSTM layers and one dense output.
Model architecture:
LSTM: 50 units, return sequences on first two layers.
Output: Dense layer for regression.
Compiles with mean squared error and Adam optimizer.​
5. Model Training and Evaluation
Trains for 100 epochs; outputs loss and validation loss for each epoch.
Uses trained model to predict on training and test sets.
Inverse transforms predictions for comparison with original values.
Evaluates model using RMSE and other metrics.​
6. Forecasting
Conducts rolling future prediction (e.g., 28-30 steps ahead), feeding predicted values iteratively.
Allows visual inspection of both historical fits and forecasting capability.​
7. Visualization
Plots original time series, train/test predictions, and future forecasts.
Matplotlib used for all graphing operations.
