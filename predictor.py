import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from train_model import DeepNN, SEQ_LEN
from data_pipeline import load_data, scale_data

# --- Configuration ---
MODEL_PATH = "model/model.pth"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/SN_ms_tot_V2.0.csv"

# --- Load model and scaler ---
scaler = joblib.load(SCALER_PATH)
model = DeepNN(input_dim=SEQ_LEN, hidden_dims=[16, 64, 8])
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Forecast Function ---
def forecast_sunspots(start_date="2020-01", years=21):
    ts = load_data(DATA_PATH)
    ts = ts.loc["1750-01":"2019-12"]
    scaled_series = scaler.transform(ts.values.reshape(-1, 1)).flatten()
    start_idx = ts.index.get_loc("2019-12-01")
    history = list(scaled_series[start_idx + 1 - SEQ_LEN : start_idx + 1])

    preds_scaled = []
    n_months = years * 12
    for _ in range(n_months):
        x = torch.tensor(history[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(x).item()
        preds_scaled.append(p)
        history.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    forecast_index = pd.date_range(start=start_date, periods=n_months, freq="MS")
    return pd.Series(preds, index=forecast_index, name="Forecasted_SSN")

# --- Main ---
if __name__ == "__main__":
    forecast = forecast_sunspots(start_date="2020-01", years=21)
    forecast.to_excel("ssn_forecast_2020_2040.xlsx")

    # Load full actual data
    full_ts = load_data(DATA_PATH)
    train = full_ts.loc["1750-01":"2008-12"]
    test = full_ts.loc["2009-01":"2019-12"]

    # Plot all together: 2020–2040 forecast only
    plt.figure(figsize=(12, 5))
    plt.plot(forecast, label="Forecast (2020–2040)", color='darkorange')
    plt.title("Sunspot Forecast (2020–2040)")
    plt.xlabel("Date")
    plt.ylabel("Smoothed Sunspot Number")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot full timeline
    plt.figure(figsize=(14, 6))
    plt.plot(train, label="Training Data (1750–2008)", color='steelblue')
    plt.plot(test, label="Test Data (2009–2019)", color='black')
    plt.plot(forecast, label="Forecast (2020–2040)", color='darkorange')
    plt.title("Sunspot Number: Train, Test & Forecast (1750–2040)")
    plt.xlabel("Date")
    plt.ylabel("Smoothed Sunspot Number")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
