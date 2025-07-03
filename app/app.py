from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# --- Load forecast Excel once ---
forecast_path = os.path.join("static", "ssn_forecast_2020_2040.xlsx")
df_forecast = pd.read_excel(forecast_path)
df_forecast["Year_Month"] = pd.to_datetime(df_forecast["Year_Month"], format="%Y-%m")

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
    return send_from_directory("static", "ssn_forecast_2020_2040.xlsx", as_attachment=True)

@app.route("/xai/<filename>")
def serve_xai_plot(filename):
    """
    Serve SHAP/IG plots stored under static/xai/ directory.
    """
    return send_from_directory("static/xai", filename)

if __name__ == "__main__":
    app.run(debug=True)