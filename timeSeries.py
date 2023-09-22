import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the first time series data
df1 = pd.read_csv('Dataset_rightSensor.csv')
df1['timestamp'] = pd.to_datetime(df1['timestamp'])  # Convert timestamp to datetime if not already

# Load the second time series data
df2 = pd.read_csv('Dataset_rightSensor_Defective.csv')
df2['timestamp'] = pd.to_datetime(df2['timestamp'])  # Convert timestamp to datetime if not already


# Function to detect anomalies in a time series and plot the results
def detect_anomalies(df, series_name):
    model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter
    model.fit(df[['value']])

    # Predict anomalies (1 for normal, -1 for anomaly)
    df['anomaly'] = model.predict(df[['value']])

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['value'], label=series_name)
    plt.scatter(df['timestamp'][df['anomaly'] == -1], df['value'][df['anomaly'] == -1], color='red', label='Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'Time Series Anomaly Detection - {series_name}')
    plt.show()


# Detect anomalies for the first time series
detect_anomalies(df1, 'Time Series 1')

# Detect anomalies for the second time series
detect_anomalies(df2, 'Time Series 2')