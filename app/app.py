from flask import Flask, render_template, request, jsonify, redirect, send_file
import pandas as pd
import requests
from io import BytesIO

app = Flask(__name__)

# GitHub raw file base URL
GITHUB_BASE_URL = "https://raw.githubusercontent.com/ameya1807/SSN_forecasting/main"

# --- Load forecast Excel from GitHub ---
def load_forecast_data():
    try:
        url = f"{GITHUB_BASE_URL}/static/ssn_forecast_2020_2040.xlsx"
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        df["Year_Month"] = pd.to_datetime(df["Year_Month"], format="%Y-%m")
        return df
    except Exception as e:
        print(f"❌ Error loading forecast data: {e}")
        return None

# Load once at startup
df_forecast = load_forecast_data()

# --- Peaks metadata ---
cycle_peaks = {
    "2023-10": {"label": "Cycle 25 Peak (Oct 2023)"},
    "2035-06": {"label": "Cycle 26 Peak (Jun 2035)"}
}

@app.route("/")
def home():
    years = list(range(2020, 2041))
    months = [(f"{i:02}", pd.to_datetime(f"2020-{i:02}-01").strftime("%B")) for i in range(1, 13)]
    return render_template("index.html", years=years, months=months, peaks=cycle_peaks)

@app.route("/api/forecast/<year_month>")
def get_forecast(year_month):
    if df_forecast is None:
        return jsonify({"error": "Forecast data not available"}), 500
    try:
        date = pd.to_datetime(year_month, format="%Y-%m")
        row = df_forecast[df_forecast["Year_Month"] == date]
        if row.empty:
            return jsonify({"error": "Date not in forecast range"}), 404
        ssn = round(float(row["Forecasted_SSN"].values[0]), 2)
        return jsonify({"ssn": ssn})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/download")
def download_forecast():
    return redirect(f"{GITHUB_BASE_URL}/static/ssn_forecast_2020_2040.xlsx")

@app.route("/xai/<filename>")
def serve_xai_plot(filename):
    try:
        # Stream image from GitHub raw URL
        image_url = f"{GITHUB_BASE_URL}/static/xai/{filename}"
        response = requests.get(image_url)
        response.raise_for_status()
        return send_file(BytesIO(response.content), mimetype="image/png")
    except Exception as e:
        return f"<h4 style='color:tomato'>Error loading image {filename}: {e}</h4>", 404

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render assigns the port via env variable
    app.run(host="0.0.0.0", port=port)
