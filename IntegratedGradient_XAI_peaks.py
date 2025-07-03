import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train_model import DeepNN, SEQ_LEN
from data_pipeline import scale_data

# --- Load model ---
model = DeepNN(input_dim=SEQ_LEN, hidden_dims=[16, 64, 8])
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()

# --- Load forecast file ---
forecast_df = pd.read_excel("ssn_forecast_2020_2040.xlsx")
forecast_df["Date"] = pd.to_datetime(forecast_df["Year_Month"], format="%Y-%m")
forecast_df.set_index("Date", inplace=True)
forecast = forecast_df["Forecasted_SSN"]

# --- Scale forecast data using original MinMaxScaler ---
from joblib import load
scaler = load("model/scaler.pkl")
forecast_scaled = scaler.transform(forecast.values.reshape(-1, 1)).flatten()

# --- Define target peaks to explain ---
target_dates = ["2023-10-01", "2035-06-01"]

def compute_integrated_gradients(input_vec, model, steps=50):
    baseline = torch.zeros_like(input_vec)
    alphas = torch.linspace(0, 1, steps).view(-1, 1)
    interpolated = baseline + alphas * (input_vec - baseline)

    interpolated.requires_grad_(True)
    attributions = []

    for i in range(steps):
        x = interpolated[i].unsqueeze(0)
        output = model(x)
        output.backward()
        grads = interpolated.grad[i]
        attributions.append(grads.detach().numpy())

    avg_grads = np.mean(np.array(attributions), axis=0)
    return (input_vec - baseline).numpy() * avg_grads

# --- Loop through each peak and plot attributions ---
for date_str in target_dates:
    target_date = pd.to_datetime(date_str)
    target_idx = forecast.index.get_loc(target_date)

    if target_idx < SEQ_LEN:
        raise ValueError(f"Not enough history before {target_date} to form input sequence.")

    input_window = forecast_scaled[target_idx - SEQ_LEN : target_idx]
    input_tensor = torch.tensor(input_window, dtype=torch.float32)

    ig = compute_integrated_gradients(input_tensor, model)

    months = pd.date_range(end=target_date, periods=SEQ_LEN, freq="MS")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.bar(months.strftime("%b %Y"), ig, color='darkcyan')
    plt.xticks(rotation=45)
    plt.title(f"Integrated Gradients Attribution for Peak: {target_date.strftime('%b %Y')}")
    plt.xlabel("Input Month")
    plt.ylabel("Contribution to Forecast")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
