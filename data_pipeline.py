# data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    df = pd.read_csv(filepath, sep=';', header=None)
    df.columns = [
        'Year', 'Month', 'Decimal_Date', 'Smoothed_SSN',
        'Standard_Deviation', 'Observations_Count', 'Definitive'
    ]
    
    # Create proper datetime index
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)

    return df[['Smoothed_SSN']]

def scale_data(ts, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    return scaled.flatten(), scaler

def prepare_sequences(series, seq_len=18):  # updated to 18
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

# Test locally
if __name__ == "__main__":
    filepath = "data/SN_ms_tot_V2.0.csv"
    ts = load_data(filepath)
    ts = ts.loc['1750-01':'2019-12']
    
    scaled_series, scaler = scale_data(ts)
    X, y = prepare_sequences(scaled_series, seq_len=18)  # updated here too
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
