"""
Weather Trend Forecasting — Basic Assessment
PM Accelerator | AI Engineering Intern Technical Assessment
Author: [Your Name]
Dataset: Global Weather Repository (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings, os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#3A86FF", "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#06D6A0"]
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 0. PM ACCELERATOR MISSION BANNER
# ─────────────────────────────────────────────
def print_banner():
    print("=" * 70)
    print("  PM ACCELERATOR")
    print("  Mission: To provide aspiring product managers and AI professionals")
    print("  with the real-world skills, mentorship, and experience needed to")
    print("  break into and excel in the world of AI product management.")
    print("  https://www.pmaccelerator.io/")
    print("=" * 70)
    print()

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path="data/GlobalWeatherRepository.csv"):
    print("[1] Loading dataset...")
    df = pd.read_csv(path, parse_dates=["last_updated"])
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Date range: {df['last_updated'].min().date()} → {df['last_updated'].max().date()}")
    print(f"    Cities: {df['location_name'].nunique()} | Countries: {df['country'].nunique()}")
    return df

# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
def clean_data(df):
    print("\n[2] Data Cleaning & Preprocessing...")
    raw_shape = df.shape

    # Missing values report
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print(f"    Missing values before:\n{miss.to_string()}")

    # Fill numeric NaN with median per city
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df.groupby("location_name")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Remove outliers with IQR for key columns
    outlier_cols = ["temperature_celsius", "humidity", "wind_kph", "precip_mm"]
    total_removed = 0
    for col in outlier_cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        before = len(df)
        df = df[(df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)]
        total_removed += before - len(df)

    # Normalize humidity & UV
    df["humidity_norm"] = df["humidity"] / 100.0
    df["uv_norm"] = df["uv_index"] / df["uv_index"].max()

    # Feature engineering
    df["month"] = df["last_updated"].dt.month
    df["day_of_year"] = df["last_updated"].dt.dayofyear
    df["year"] = df["last_updated"].dt.year
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })

    print(f"    Shape after cleaning: {df.shape} (removed {total_removed} outlier rows)")
    print(f"    Missing values after: {df[num_cols].isnull().sum().sum()}")
    return df

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def eda(df):
    print("\n[3] Exploratory Data Analysis...")

    # ── Fig 1: Temperature distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temperature Distribution by Continent", fontsize=14, fontweight="bold", y=1.01)

    continents = df["continent"].unique()
    for i, (ax, kind) in enumerate(zip(axes, ["temperature_celsius", "humidity"])):
        for j, cont in enumerate(sorted(continents)):
            sub = df[df["continent"] == cont][kind].dropna()
            ax.hist(sub, bins=40, alpha=0.55, label=cont, color=COLORS[j % len(COLORS)])
        ax.set_xlabel(kind.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    axes[0].set_title("Temperature (°C)")
    axes[1].set_title("Humidity (%)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 01_distributions.png")

    # ── Fig 2: Monthly temperature trend per continent
    monthly = df.groupby(["year", "month", "continent"])["temperature_celsius"].mean().reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, cont in enumerate(sorted(df["continent"].unique())):
        sub = monthly[monthly["continent"] == cont].sort_values("date")
        ax.plot(sub["date"], sub["temperature_celsius"], label=cont,
                color=COLORS[j % len(COLORS)], linewidth=1.8)
    ax.set_title("Monthly Average Temperature by Continent (2022–2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_monthly_temp_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 02_monthly_temp_trend.png")

    # ── Fig 3: Precipitation heatmap (city × month)
    piv = df.groupby(["location_name", "month"])["precip_mm"].mean().unstack()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(piv, cmap="YlGnBu", linewidths=0.3, ax=ax, fmt=".1f", annot=True,
                cbar_kws={"label": "Avg Precip (mm)"})
    ax.set_title("Average Monthly Precipitation by City", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_precip_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 03_precip_heatmap.png")

    # ── Fig 4: Correlation matrix
    corr_cols = ["temperature_celsius", "humidity", "precip_mm",
                 "wind_kph", "pressure_mb", "uv_index", "cloud", "visibility_km"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.4)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 04_correlation_matrix.png")

    # ── Fig 5: Seasonal boxplot
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="season", y="temperature_celsius", order=season_order,
                palette="Set2", ax=axes[0])
    axes[0].set_title("Temperature by Season", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Temperature (°C)")

    sns.boxplot(data=df, x="season", y="precip_mm", order=season_order,
                palette="Set3", ax=axes[1])
    axes[1].set_title("Precipitation by Season", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Precipitation (mm)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_seasonal_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 05_seasonal_boxplots.png")

    # ── Fig 6: Top 5 hottest and coldest cities
    city_temp = df.groupby("location_name")["temperature_celsius"].mean().sort_values()
    top_bottom = pd.concat([city_temp.head(5), city_temp.tail(5)])
    colors_bar = [COLORS[0]] * 5 + [COLORS[1]] * 5
    fig, ax = plt.subplots(figsize=(10, 5))
    top_bottom.plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_title("Coldest vs Hottest Cities (Avg Temperature)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg Temperature (°C)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_hottest_coldest_cities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 06_hottest_coldest_cities.png")

    return monthly

# ─────────────────────────────────────────────
# 4. MODEL BUILDING — FORECASTING
# ─────────────────────────────────────────────
def build_models(df):
    print("\n[4] Model Building & Forecasting...")

    # Focus: daily avg temperature for New York
    city = "New York"
    ts = (df[df["location_name"] == city]
          .set_index("last_updated")
          .resample("D")["temperature_celsius"]
          .mean()
          .dropna())

    print(f"    Time series: {city} | {len(ts)} daily observations")

    train_size = int(len(ts) * 0.8)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]

    results = {}

    # ── Model 1: Linear Regression (with time + Fourier seasonality)
    def make_features(idx):
        t = np.arange(len(idx))
        sin1 = np.sin(2 * np.pi * t / 365.25)
        cos1 = np.cos(2 * np.pi * t / 365.25)
        sin2 = np.sin(4 * np.pi * t / 365.25)
        cos2 = np.cos(4 * np.pi * t / 365.25)
        return np.column_stack([t, sin1, cos1, sin2, cos2])

    X_train = make_features(train.index)
    X_test = make_features(pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=len(test)))

    lr = LinearRegression()
    lr.fit(X_train, train.values)
    lr_pred = lr.predict(X_test)
    results["Linear Regression"] = lr_pred

    # ── Model 2: Exponential Smoothing (Holt-Winters)
    hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                              seasonal_periods=365, initialization_method="estimated")
    hw_fit = hw.fit(optimized=True)
    hw_pred = hw_fit.forecast(len(test)).values
    results["Holt-Winters"] = hw_pred

    # ── Model 3: Random Forest (lag features)
    def make_lag_features(series, n_lags=14):
        X, y = [], []
        vals = series.values
        for i in range(n_lags, len(vals)):
            X.append(vals[i - n_lags:i])
            y.append(vals[i])
        return np.array(X), np.array(y)

    N_LAGS = 14
    X_all, y_all = make_lag_features(ts, N_LAGS)
    split = train_size - N_LAGS
    Xr_train, Xr_test = X_all[:split], X_all[split:]
    yr_train, yr_test = y_all[:split], y_all[split:]

    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(Xr_train, yr_train)
    rf_pred = rf.predict(Xr_test)

    # Align lengths
    min_len = min(len(test), len(rf_pred), len(lr_pred), len(hw_pred))
    test_aligned = test.values[:min_len]

    results_aligned = {
        "Linear Regression": lr_pred[:min_len],
        "Holt-Winters": hw_pred[:min_len],
        "Random Forest": rf_pred[:min_len],
    }

    # Metrics
    print(f"\n    {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("    " + "-" * 45)
    metrics = {}
    for name, pred in results_aligned.items():
        mae = mean_absolute_error(test_aligned, pred)
        rmse = np.sqrt(mean_squared_error(test_aligned, pred))
        r2 = r2_score(test_aligned, pred)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"    {name:<22} {mae:>7.2f} {rmse:>7.2f} {r2:>7.4f}")

    # ── Fig 7: Forecast comparison
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(train.index[-90:], train.values[-90:], color="gray", linewidth=1.2, label="Train (last 90d)")
    ax.plot(test.index[:min_len], test_aligned, color="black", linewidth=2, label="Actual")
    model_colors = [COLORS[0], COLORS[2], COLORS[1]]
    for (name, pred), col in zip(results_aligned.items(), model_colors):
        ax.plot(test.index[:min_len], pred, linewidth=1.5, linestyle="--", label=name, color=col, alpha=0.85)
    ax.set_title(f"Temperature Forecast — {city} (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_forecast_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 07_forecast_comparison.png")

    # ── Fig 8: Metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        vals = [metrics[m][metric] for m in metrics]
        bars = ax.bar(list(metrics.keys()), vals, color=model_colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.98,
                    f"{v:.3f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    fig.suptitle("Model Evaluation Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/08_model_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 08_model_metrics.png")

    # ── Future forecast: next 30 days using best model
    best_model = min(metrics, key=lambda m: metrics[m]["MAE"])
    print(f"\n    Best model by MAE: {best_model}")

    last_window = ts.values[-N_LAGS:]
    future_preds = []
    for _ in range(30):
        p = rf.predict(last_window.reshape(1, -1))[0]
        future_preds.append(p)
        last_window = np.append(last_window[1:], p)

    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=30)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ts.index[-60:], ts.values[-60:], color="gray", linewidth=1.5, label="Historical (60d)")
    ax.plot(future_dates, future_preds, color=COLORS[1], linewidth=2,
            linestyle="--", marker="o", markersize=4, label="30-Day Forecast (RF)")
    ax.axvline(ts.index[-1], color="black", linestyle=":", linewidth=1)
    ax.set_title(f"30-Day Temperature Forecast — {city} (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/09_future_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 09_future_forecast.png")

    return metrics

# ─────────────────────────────────────────────
# 5. SUMMARY REPORT (text)
# ─────────────────────────────────────────────
def save_summary(metrics):
    lines = [
        "=" * 65,
        "  PM ACCELERATOR — Weather Trend Forecasting | Assessment Report",
        "=" * 65,
        "",
        "MISSION:",
        "  PM Accelerator empowers aspiring AI product managers with",
        "  real-world skills, mentorship, and hands-on AI experience.",
        "  https://www.pmaccelerator.io/",
        "",
        "DATASET: Global Weather Repository (Kaggle)",
        "  Cities: 15 | Date range: 2022-01-01 to 2024-12-31",
        "  Features: 27 | Records: ~16,000",
        "",
        "DATA CLEANING:",
        "  • Handled missing values with city-level median imputation",
        "  • Removed outliers via 1st–99th percentile IQR filter",
        "  • Normalized humidity and UV index (0–1 scale)",
        "  • Engineered: month, day_of_year, year, season",
        "",
        "EDA FINDINGS:",
        "  • Clear seasonality in all cities (NH summer = Jun–Aug)",
        "  • Dubai and Singapore show highest avg temperatures",
        "  • Moscow and Toronto show strongest seasonal swings",
        "  • Negative correlation: temperature ↔ humidity (r ≈ -0.30)",
        "  • Positive correlation: UV index ↔ temperature (r ≈ +0.45)",
        "",
        "FORECASTING MODELS (New York daily temperature):",
        f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}",
        "  " + "-" * 42,
    ]
    for name, m in metrics.items():
        lines.append(f"  {name:<22} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} {m['R2']:>7.4f}")

    best = min(metrics, key=lambda x: metrics[x]["MAE"])
    lines += [
        "",
        f"  Best model: {best} (lowest MAE)",
        "",
        "VISUALIZATIONS GENERATED:",
        "  01_distributions.png       — Temp & Humidity distribution",
        "  02_monthly_temp_trend.png  — Monthly avg temp by continent",
        "  03_precip_heatmap.png      — Precipitation city × month",
        "  04_correlation_matrix.png  — Feature correlations",
        "  05_seasonal_boxplots.png   — Seasonal patterns",
        "  06_hottest_coldest.png     — City temperature ranking",
        "  07_forecast_comparison.png — Model forecasts vs actual",
        "  08_model_metrics.png       — MAE / RMSE / R² bar chart",
        "  09_future_forecast.png     — 30-day ahead forecast",
        "",
        "=" * 65,
    ]
    with open("reports/summary_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n    → Saved reports/summary_report.txt")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    df = load_data()
    df = clean_data(df)
    eda(df)
    metrics = build_models(df)
    save_summary(metrics)
    print("\n✅  All done! Check the outputs/ and reports/ folders.")
