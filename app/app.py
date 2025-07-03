from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import requests
from io import BytesIO

app = Flask(__name__)

# GitHub raw file base URL
GITHUB_BASE_URL = "https://raw.githubusercontent.com/ameya1807/SSN_forecasting/main"

# --- Load forecast Excel from GitHub ---
def load_forecast_data():
    try:
        excel_url = f"{GITHUB_BASE_URL}/static/ssn_forecast_2020_2040.xlsx"
        response = requests.get(excel_url)
        response.raise_for_status()
        
        df_forecast = pd.read_excel(BytesIO(response.content))
        df_forecast["Year_Month"] = pd.to_datetime(df_forecast["Year_Month"], format="%Y-%m")
        return df_forecast
    except Exception as e:
        print(f"Error loading forecast data: {e}")
        return None

# Load data once at startup
df_forecast = load_forecast_data()

# --- Cycle peaks (for dropdowns) ---
cycle_peaks = {
    "2023-10": {"label": "Cycle 25 Peak (Oct 2023)"},
    "2035-06": {"label": "Cycle 26 Peak (Jun 2035)"}
}

@app.route("/")
def home():
    years = list(range(2020, 2041))
    months = [
        ("01", "January"), ("02", "February"), ("03", "March"), ("04", "April"),
        ("05", "May"), ("06", "June"), ("07", "July"), ("08", "August"),
        ("09", "September"), ("10", "October"), ("11", "November"), ("12", "December")
    ]
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
    """Redirect to GitHub raw file for download"""
    excel_url = f"{GITHUB_BASE_URL}/static/ssn_forecast_2020_2040.xlsx"
    return redirect(excel_url)

@app.route("/xai/<filename>")
def serve_xai_plot(filename):
    """Redirect to GitHub raw file for XAI plots"""
    plot_url = f"{GITHUB_BASE_URL}/static/xai/{filename}"
    return redirect(plot_url)

if __name__ == "__main__":
    app.run(debug=True)
