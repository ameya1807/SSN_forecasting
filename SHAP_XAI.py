import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train_model import DeepNN, SEQ_LEN
from data_pipeline import scale_data
from joblib import load

# --- Load model ---
model = DeepNN(input_dim=SEQ_LEN, hidden_dims=[16, 64, 8])
model.load_state_dict(torch.load("model/model.pth", map_location="cpu", weights_only=True))
model.eval()

# --- Define a wrapper for SHAP that handles NumPy input ---
def model_forward(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x).squeeze(1)

# --- Load forecast ---
forecast_df = pd.read_excel("ssn_forecast_2020_2040.xlsx")
forecast_df["Date"] = pd.to_datetime(forecast_df["Year_Month"], format="%Y-%m")
forecast_df.set_index("Date", inplace=True)
forecast = forecast_df["Forecasted_SSN"]

# --- Load scaler and scale forecast ---
scaler = load("model/scaler.pkl")
forecast_scaled = scaler.transform(forecast.values.reshape(-1, 1)).flatten()

# --- Define target peaks ---
peak_dates = {
    "2023-10-01": "Cycle 25 Peak",
    "2035-06-01": "Cycle 26 Peak"
}

# --- Build background dataset for SHAP ---
target_1_idx = forecast.index.get_loc(pd.to_datetime("2023-10-01"))
X_background = []

for i in range(target_1_idx - 100 - SEQ_LEN, target_1_idx - SEQ_LEN):
    window = forecast_scaled[i : i + SEQ_LEN]
    if len(window) == SEQ_LEN:
        X_background.append(window)

X_background = np.array(X_background, dtype=np.float32)

# --- Create SHAP explainer ---
explainer = shap.Explainer(model_forward, X_background)

# --- Loop through peaks and explain ---
for peak_date_str, label in peak_dates.items():
    peak_date = pd.to_datetime(peak_date_str)
    peak_idx = forecast.index.get_loc(peak_date)

    input_window = forecast_scaled[peak_idx - SEQ_LEN : peak_idx]
    input_tensor = np.array(input_window, dtype=np.float32).reshape(1, -1)

    # Compute SHAP values
    shap_values = explainer(input_tensor)
    shap_vals = shap_values.values[0]  # already NumPy

    # Create date labels for input months
    months = pd.date_range(end=peak_date, periods=SEQ_LEN, freq="MS")

    # Plot SHAP values
    plt.figure(figsize=(10, 5))
    plt.bar(months.strftime("%b %Y"), shap_vals, color='teal')
    plt.xticks(rotation=45)
    plt.title(f"SHAP Attribution for Forecast: {label} ({peak_date.strftime('%b %Y')})")
    plt.xlabel("Input Month")
    plt.ylabel("SHAP Value (Impact on Forecast)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
