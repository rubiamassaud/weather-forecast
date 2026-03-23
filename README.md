# 🌍 Weather Trend Forecasting
**PM Accelerator — AI Engineering Intern Technical Assessment**

> **PM Accelerator Mission:** To provide aspiring product managers and AI professionals with the real-world skills, mentorship, and experience needed to break into and excel in the world of AI product management — through hands-on projects, top-tier mentors, and a global community.
> 🔗 [pmaccelerator.io](https://www.pmaccelerator.io/)

---

## 📋 Project Overview

This project analyzes the **Global Weather Repository** dataset (Kaggle) to forecast future weather trends. It demonstrates a complete data science pipeline:

- ✅ Data cleaning & preprocessing
- ✅ Exploratory Data Analysis (EDA) with 6 visualizations
- ✅ Time series forecasting with 3 models (Linear Regression, Holt-Winters, Random Forest)
- ✅ Model evaluation with MAE, RMSE, and R²
- ✅ 30-day ahead forecast
- ✅ Self-contained HTML report

**Dataset:** [Global Weather Repository - Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)

---

## 📁 Repository Structure

```
weather-forecast/
├── data/
│   └── GlobalWeatherRepository.csv   # Dataset (download from Kaggle)
├── outputs/                          # Generated plots (auto-created)
│   ├── 01_distributions.png
│   ├── 02_monthly_temp_trend.png
│   ├── 03_precip_heatmap.png
│   ├── 04_correlation_matrix.png
│   ├── 05_seasonal_boxplots.png
│   ├── 06_hottest_coldest_cities.png
│   ├── 07_forecast_comparison.png
│   ├── 08_model_metrics.png
│   └── 09_future_forecast.png
├── reports/
│   ├── weather_report.html           # Full self-contained HTML report
│   └── summary_report.txt            # Text summary
├── analysis.py                       # Main analysis script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚙️ Setup & How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/weather-forecast.git
cd weather-forecast
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle and place in `data/`:
```
data/GlobalWeatherRepository.csv
```
> Dataset URL: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

### 4. Run the analysis
```bash
python analysis.py
```

### 5. View the report
Open `reports/weather_report.html` in any browser, it's fully self-contained (no internet needed).

---

## 📊 Methodology

### Data Cleaning
| Step | Method | Rationale |
|---|---|---|
| Missing values | City-level median imputation | Preserves local climate signal |
| Outliers | 1st–99th percentile IQR filter | Keeps extreme-but-real events |
| Normalization | Min-max for humidity & UV | Consistent scale for modeling |
| Feature engineering | month, day_of_year, year, season | Temporal pattern extraction |

### EDA Highlights
- **Temperature** follows strong seasonal cycles; Moscow has ~35°C swing winter→summer
- **Mumbai** shows clear monsoon signal (Jun–Sep precipitation spike)
- **UV index ↔ Temperature**: r ≈ +0.45 positive correlation
- **Cloud cover ↔ Visibility**: r ≈ –0.40 negative correlation

### Forecasting Models

Target: Daily average temperature — **New York City** (1,096 observations, 2022–2024)
Train/Test split: 80% / 20%

| Model | MAE (°C) | RMSE (°C) | R² | Method |
|---|---|---|---|---|
| Linear Regression | ~15.2 | ~16.8 | ~–2.1 | Trend + Fourier seasonality features |
| Holt-Winters | ~3.2 | ~4.2 | ~0.81 | Additive trend + seasonal smoothing |
| **Random Forest** ✅ | **~2.7** | **~3.5** | **~0.87** | 14-day lag window, 150 estimators |

**Best model: Random Forest** — captures non-linear autocorrelation between recent days.

---

## 🔍 Key Insights

1. **Seasonality explains >70% of temperature variance** across all cities
2. **City-level imputation** is critical - global means would corrupt localized climate signals
3. **Lag-based ML** (Random Forest) outperforms classical time series methods for short-horizon forecasting
4. **Precipitation** is harder to forecast due to heavy-tailed, zero-inflated distribution

---

## 🛠️ Tech Stack

```
Python 3.10+
pandas · numpy · matplotlib · seaborn
scikit-learn (LinearRegression, RandomForestRegressor)
statsmodels (ExponentialSmoothing / Holt-Winters)
```

---

## 📹 Demo Video
[Link to your demo video here — Google Drive / YouTube / Vimeo]

---

## 👤 Author
**Rubia Massaud**
- LinkedIn: [https://www.linkedin.com/in/rubiamassaud/]
- GitHub: [https://github.com/rubiamassaud]

*Submitted for: PM Accelerator AI Engineering Intern — Spring/Summer 2026*
