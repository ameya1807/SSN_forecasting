# SSN_forecasting
# ☀️ Sunspot Forecasting & Explainability App

This project forecasts monthly smoothed sunspot numbers from 2020–2040 using a deep learning model, with built-in explainability using SHAP and Integrated Gradients (IG). It includes a responsive Flask web interface with dropdown selection, downloadable forecast, and XAI visualizations.

---

## 🚀 Features

- 📈 Forecasts sunspot numbers from 2020 to 2040
- 💾 Forecast data downloadable as Excel
- 📊 SHAP & Integrated Gradients explanations for cycle peaks
- 🔍 Interactive dropdowns to explore monthly forecasts
- 🎨 Dark-themed Bootstrap-styled UI
- ☁️ Hosted forecast and plots directly from GitHub

---

## 🧠 ML Model

- **Architecture**: 3-layer feedforward neural network
- **Input**: Last 18 months of smoothed SSN
- **Output**: Forecasted SSN for next month
- **Training**: Data from 1750–2008 (train), 2009–2019 (validation)
- **Prediction**: Autoregressive inference for 2020–2040
- **Uncertainty**: Optional conformal prediction (commented in production)
- **XAI**: SHAP + IG for peak cycles:  
  - Cycle 25 peak (Oct 2023)  
  - Cycle 26 peak (Jun 2035)

---

## 🌐 App Preview

- `app.py`: Backend API and routing
- `index.html`: Bootstrap UI with dropdowns and XAI options
- `/static/xai/`: Folder for XAI plots (`shap_2023-10.png`, `ig_2035-06.png`, etc.)

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ameya1807/SSN_forecasting.git
cd SSN_forecasting
pip install -r requirements.txt

python app/app.py

http://127.0.0.1:5000

SSN_forecasting/
│
├── app/
│   └── app.py
│   └── templates/index.html
│
├── data/                    # SILSO source data
│
├── model/                  # Trained model weights + scaler
│
├── static/
│   ├── ssn_forecast_2020_2040.xlsx
│   └── xai/
│       ├── shap_2023-10.png
│       ├── ig_2023-10.png
│       └── ...
│
├── train_model.py
├── predictor.py
├── SHAP_XAI.py
├── IntegratedGradient_XAI_peaks.py
├── compare_ig_rise_rate.py
├── requirements.txt
└── README.md

📚 Credits
Data source: SILSO – World Data Center - https://www.sidc.be/SILSO/datafiles - 13-month smoothed monthly total sunspot number [1/1749 - now] 

SHAP: Lundberg & Lee (2017)

Integrated Gradients: Sundararajan et al. (2017)

Built by Ameya
IIT BHU 2023-28
