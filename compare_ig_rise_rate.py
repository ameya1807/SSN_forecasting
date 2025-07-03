import pandas as pd
import torch
import numpy as np
from joblib import load
from train_model import DeepNN, SEQ_LEN
from data_pipeline import scale_data

# --- Load model ---
model = DeepNN(input_dim=SEQ_LEN, hidden_dims=[16, 64, 8])
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()

# --- Load forecast data ---
forecast_df = pd.read_excel("ssn_forecast_2020_2040.xlsx")
forecast_df["Date"] = pd.to_datetime(forecast_df["Year_Month"], format="%Y-%m")
forecast_df.set_index("Date", inplace=True)
forecast = forecast_df["Forecasted_SSN"]

# --- Load scaler and scale forecast ---
scaler = load("model/scaler.pkl")
forecast_scaled = scaler.transform(forecast.values.reshape(-1, 1)).flatten()

# --- Define target peaks and window ---
target_dates = {
    "2023-10-01": "Cycle 25 Peak",
    "2035-06-01": "Cycle 26 Peak"
}
window_months = 6

# --- Helper: compute IG ---
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

# --- Print rise rate and IG attribution for each peak ---
for date_str, label in target_dates.items():
    peak_date = pd.to_datetime(date_str)
    peak_idx = forecast.index.get_loc(peak_date)

    # Rise rate from forecast
    ssn_peak = forecast.iloc[peak_idx]
    ssn_prev = forecast.iloc[peak_idx - window_months]
    rise_rate = (ssn_peak - ssn_prev) / window_months

    # IG input and attribution
    input_window = forecast_scaled[peak_idx - SEQ_LEN : peak_idx]
    input_tensor = torch.tensor(input_window, dtype=torch.float32)
    ig = compute_integrated_gradients(input_tensor, model)
    ig_sum = np.sum(ig[-window_months:])

    print(f"🔍 {label} ({peak_date.strftime('%b %Y')})")
    print(f"Rise rate over last {window_months} months: {rise_rate:.3f}")
    print(f"Sum of IG attributions over last {window_months} months: {ig_sum:.3f}\n")
