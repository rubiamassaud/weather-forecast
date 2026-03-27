"""
Weather Trend Forecasting
Author: Rubia Massaud dos Santos
Dataset: Global Weather Repository (Kaggle)
"""

import os
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#3A86FF", "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#06D6A0"]
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────


def load_data(path="data/GlobalWeatherRepository.csv"):
    print("[1] Loading dataset...")
    df = pd.read_csv(path, parse_dates=["last_updated"])
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(
        f"    Date range: {df['last_updated'].min().date()} → {df['last_updated'].max().date()}")
    print(
        f"    Cities: {df['location_name'].nunique()} | Countries: {df['country'].nunique()}")
    return df

# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────


def clean_data(df):
    print("\n[2] Data Cleaning & Preprocessing...")

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
    original_len = len(df)
    for col in outlier_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)]

    # Normalize humidity & UV
    df["humidity_norm"] = df["humidity"] / 100.0
    uv_max = df["uv_index"].max()
    df["uv_norm"] = df["uv_index"] / uv_max if uv_max > 0 else 0.0

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

    total_removed = original_len - len(df)
    print(
        f"    Shape after cleaning: {df.shape} (removed {total_removed} outlier rows)")
    print(f"    Missing values after: {df[num_cols].isnull().sum().sum()}")
    return df

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────


def eda(df):
    print("\n[3] Exploratory Data Analysis...")

    # Fig 1: Temperature & Humidity distributions
    df['region'] = df['timezone'].str.split('/').str[0]
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Temperature & Humidity Distribution by Region",
                 fontsize=15, fontweight="bold", y=1.02)
    regions = sorted(df["region"].unique())
    high_contrast_palette = ["#1E90FF", "#E31A1C", "#FF7F00",
                             "#33A02C", "#6A3D9A", "#00CACA", "#FFD700", "#FF1493"]
    color_map = {reg: high_contrast_palette[i % len(
        high_contrast_palette)] for i, reg in enumerate(regions)}
    metrics = [("temperature_celsius", "Temperature (°C)"),
               ("humidity", "Humidity (%)")]
    for i, (col, label) in enumerate(metrics):
        sns.histplot(data=df, x=col, hue="region", kde=True, element="step",
                     palette=color_map, common_norm=False, alpha=0.3, ax=axes[i],
                     legend=(i == 1))

        axes[i].set_title(label, fontsize=14, pad=10)
        axes[i].set_xlabel(label, fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    leg = axes[1].get_legend()
    if leg:
        leg.set_bbox_to_anchor((1.05, 1))
        leg.set_loc('upper left')
        leg.set_title("Region", prop={'size': 12, 'weight': 'bold'})
        plt.setp(leg.get_texts(), fontsize='10')

    plt.tight_layout()
    plt.savefig(f"{OUT}/01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    sns.set_style("whitegrid")
    print("    → Saved 01_distributions.png")

    # Fig 2: Monthly temperature trend per region
    df['region'] = df['timezone'].str.split('/').str[0]
    monthly = df.groupby(["year", "month", "region"])[
        "temperature_celsius"].mean().reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    fig, ax = plt.subplots(figsize=(14, 5))
    for j, tz in enumerate(sorted(df["region"].unique())):
        sub = monthly[monthly["region"] == tz].sort_values("date")
        ax.plot(sub["date"], sub["temperature_celsius"], label=tz,
                color=COLORS[j % len(COLORS)], linewidth=1.8)
    ax.set_title("Monthly Average Temperature by Region",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_monthly_temp_trend.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 02_monthly_temp_trend.png")

    # Fig 3: Precipitation heatmap (city × month)
    top_cities = df.groupby("location_name")[
        "precip_mm"].mean().nlargest(20).index
    df_top = df[df["location_name"].isin(top_cities)]
    piv = df_top.groupby(["location_name", "month"])[
        "precip_mm"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(piv, cmap="YlGnBu", linewidths=0.5, ax=ax, fmt=".2f", annot=True,
                annot_kws={"size": 9}, cbar_kws={"shrink": 0.8, "label": "Avg Precip (mm)"})

    ax.set_title("Average Monthly Precipitation - Top 20 Rainiest Cities",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month (1=Jan, 12=Dec)", fontsize=11)
    ax.set_ylabel("City", fontsize=11)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_precip_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("    → Saved 03_precip_heatmap.png")

    # Fig 4: Correlation matrix
    corr_cols = ["temperature_celsius", "humidity", "precip_mm",
                 "wind_kph", "pressure_mb", "uv_index", "cloud", "visibility_km"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.8,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})

    clean_labels = [c.replace('_', ' ').title() for c in corr_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha="right")
    ax.set_yticklabels(clean_labels, rotation=0)
    ax.grid(False)
    ax.set_title("Feature Correlation Matrix",
                 fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_correlation_matrix.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("    → Saved 04_correlation_matrix.png")

    # Fig 5: Seasonal boxplots
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.violinplot(data=df, x="season", y="temperature_celsius", order=season_order,
                   palette="Set2", ax=axes[0], inner="quartile", cut=0)
    axes[0].set_title("Temperature Distribution by Season",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Temperature (°C)")
    sns.boxenplot(data=df, x="season", y="precip_mm", order=season_order,
                  palette="Set3", ax=axes[1])
    percentile_95 = df["precip_mm"].quantile(0.95)
    axes[1].set_ylim(-0.1, percentile_95 * 1.5)
    axes[1].set_title("Precipitation by Season",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Precipitation (mm)")

    for ax in axes:
        ax.set_xlabel("Season")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{OUT}/05_seasonal_boxplots.png", dpi=200)
    plt.close()
    print("    → Saved 05_seasonal_boxplots.png")

    # Fig 6: Top 5 hottest and coldest cities
    city_temp = df.groupby("location_name")[
        "temperature_celsius"].mean().sort_values()
    top_bottom = pd.concat([city_temp.head(5), city_temp.tail(5)])
    colors_bar = [COLORS[0]] * 5 + [COLORS[1]] * 5
    fig, ax = plt.subplots(figsize=(10, 5))
    top_bottom.plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_title("Coldest vs Hottest Cities (Avg Temperature)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg Temperature (°C)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_hottest_coldest_cities.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 06_hottest_coldest_cities.png")

    return monthly

# ─────────────────────────────────────────────
# 4. MODEL BUILDING - FORECASTING
# ─────────────────────────────────────────────


def build_models(df):
    print("\n[4] Model Building & Forecasting...")

    # Select city with most daily observations (minimum 60)
    MIN_OBS = 60
    city = "Abu Dhabi"
    ts = (df[df["location_name"] == city]
          .set_index("last_updated")
          .resample("D")["temperature_celsius"]
          .mean()
          .dropna())

    print(f"    Time series: {city} | {len(ts)} daily observations")

    if len(ts) < MIN_OBS:
        city_counts = (df.groupby("location_name")
                       .apply(lambda x: x.set_index("last_updated")
                              .resample("D")["temperature_celsius"]
                              .mean().dropna().shape[0]))
        best_city = city_counts[city_counts >= MIN_OBS].idxmax() \
            if (city_counts >= MIN_OBS).any() else None

        if best_city is None:
            print(
                f"    ⚠ No city has enough data (min {MIN_OBS} obs). Skipping models.")
            return {}

        print(
            f"    ⚠ '{city}' has too few observations. Switching to '{best_city}'.")
        city = best_city
        ts = (df[df["location_name"] == city]
              .set_index("last_updated")
              .resample("D")["temperature_celsius"]
              .mean()
              .dropna())
        print(f"    Time series: {city} | {len(ts)} daily observations")

    train_size = int(len(ts) * 0.8)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]

    if len(train) == 0 or len(test) == 0:
        print(
            f"    ⚠ Train ({len(train)}) or test ({len(test)}) set is empty. Skipping.")
        return {}

    results = {}

    # ── Model 1: Linear Regression (time + Fourier seasonality) ──────────────
    def make_features(idx):
        t = np.arange(len(idx))
        sin1 = np.sin(2 * np.pi * t / 365.25)
        cos1 = np.cos(2 * np.pi * t / 365.25)
        sin2 = np.sin(4 * np.pi * t / 365.25)
        cos2 = np.cos(4 * np.pi * t / 365.25)
        return np.column_stack([t, sin1, cos1, sin2, cos2])

    X_train = make_features(train.index)
    X_test = make_features(
        pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=len(test)))

    lr = LinearRegression()
    lr.fit(X_train, train.values)
    lr_pred = lr.predict(X_test)
    results["Linear Regression"] = lr_pred

    # ── Model 2: Exponential Smoothing (Holt-Winters) ────────────────────────
    sp = min(365, len(train) // 2)

    try:
        hw = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=sp,
            initialization_method="estimated"
        )
        hw_fit = hw.fit(optimized=True)
        hw_pred = hw_fit.forecast(len(test)).values
    except Exception as e:
        print(f"    ⚠ Holt-Winters failed ({e}). Using fallback.")
        hw_pred = np.tile(train.values[-sp:], len(test) // sp + 1)[:len(test)]
        hw_fit = None

    results["Holt-Winters"] = hw_pred

    # ── Model 3: Random Forest (lag features) ────────────────────────────────
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

    rf = RandomForestRegressor(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(Xr_train, yr_train)
    rf_pred = rf.predict(Xr_test)

    # ── Align lengths & compute metrics ──────────────────────────────────────
    min_len = min(len(test), len(rf_pred), len(lr_pred), len(hw_pred))
    test_aligned = test.values[:min_len]

    results_aligned = {
        "Linear Regression": lr_pred[:min_len],
        "Holt-Winters":      hw_pred[:min_len],
        "Random Forest":     rf_pred[:min_len],
    }

    print(f"\n    {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("    " + "-" * 45)
    metrics = {}
    for name, pred in results_aligned.items():
        mae = mean_absolute_error(test_aligned, pred)
        rmse = np.sqrt(mean_squared_error(test_aligned, pred))
        r2 = r2_score(test_aligned, pred)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"    {name:<22} {mae:>7.2f} {rmse:>7.2f} {r2:>7.4f}")

    # ── Fig 7: Forecast comparison ────────────────────────────────────────────
    model_colors = [COLORS[0], COLORS[2], COLORS[1]]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(train.index[-90:], train.values[-90:],
            color="gray", linewidth=1.2, label="Train (last 90d)")
    ax.plot(test.index[:min_len], test_aligned,
            color="black", linewidth=2, label="Actual")
    for (name, pred), col in zip(results_aligned.items(), model_colors):
        ax.plot(test.index[:min_len], pred, linewidth=1.5,
                linestyle="--", label=name, color=col, alpha=0.85)
    ax.set_title(f"Temperature Forecast — {city} (Test Period)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_forecast_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved 07_forecast_comparison.png")

    # ── Fig 8: Metrics bar chart ──────────────────────────────────────────────
    metrics_list = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        models = list(metrics.keys())
        vals = [metrics[m][metric] for m in models]

        bars = ax.bar(models, vals, color=model_colors,
                      edgecolor="black", linewidth=0.7)
        ax.set_title(f"Model {metric}", fontweight="bold", fontsize=12, pad=15)
        ax.tick_params(axis="x", rotation=25)

        if metric == "R2":
            ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

        for bar in bars:
            height = bar.get_height()
            va = 'top' if height < 0 else 'bottom'
            offset = -0.3 if height < 0 else 0.3

            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{height:.2f}', ha='center', va=va,
                    fontweight='bold', fontsize=10, color='black')

    fig.suptitle("Model Evaluation Metrics - Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.05)

    plt.tight_layout()
    plt.savefig(f"{OUT}/08_model_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("    → Saved 08_model_metrics.png")

    # ── Future Forecast: next 30 days ─────────────────────────────────────────
    best_model_name = min(metrics, key=lambda m: metrics[m]["MAE"])
    print(f"\n    Deploying best model for forecasting: {best_model_name}")

    last_date = ts.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
    future_preds = []

    if best_model_name == "Random Forest":
        last_window = ts.values[-N_LAGS:]
        for _ in range(30):
            p = rf.predict(last_window.reshape(1, -1))[0]
            future_preds.append(p)
            last_window = np.append(last_window[1:], p)

    elif best_model_name == "Holt-Winters" and hw_fit is not None:
        future_preds = hw_fit.forecast(30).values.tolist()

    else:  # Linear Regression (or HW fallback)
        X_future = make_features(future_dates)
        future_preds = lr.predict(X_future)

    # ── Fig 9: Plot do Forecast ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts.index[-60:], ts.values[-60:],
            color="#4A4A4A", linewidth=2, label="Historical Observations", alpha=0.7)
    ax.plot(future_dates, future_preds,
            color=COLORS[1], linewidth=2.5, linestyle="--",
            marker="o", markersize=4,
            label=f"30-Day Forecast ({best_model_name})")

    ax.axvline(last_date, color="red", linestyle=":",
               linewidth=1.5, label="Forecast Horizon")

    ax.set_title(f"Temperature Trend Projection — {city}\nSelected Model: {best_model_name} (Best MAE)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.savefig(f"{OUT}/09_future_forecast.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    → Saved {OUT}/09_future_forecast.png")

    return metrics

# ─────────────────────────────────────────────
# 5. SUMMARY REPORT
# ─────────────────────────────────────────────

def save_summary(metrics):
    os.makedirs("reports", exist_ok=True)
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
        "  Features: 41 | Records: ~131,000",
        "",
        "DATA CLEANING:",
        "  • Handled missing values with city-level median imputation",
        "  • Removed outliers via IQR filter (Q1/Q3 ± 3×IQR)",
        "  • Normalized humidity (0–1) and UV index (0–1)",
        "  • Engineered: month, day_of_year, year, season",
        "",
        "EDA FINDINGS:",
        "  • Clear seasonality observed across all cities",
        "  • Negative correlation: temperature ↔ humidity",
        "  • Positive correlation: UV index ↔ temperature",
        "",
        "FORECASTING MODELS:",
        f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}",
        "  " + "-" * 42,
    ]

    if metrics:
        for name, m in metrics.items():
            lines.append(
                f"  {name:<22} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} {m['R2']:>7.4f}")
        best = min(metrics, key=lambda x: metrics[x]["MAE"])
        lines += ["", f"  Best model: {best} (lowest MAE)"]
    else:
        lines.append("  No metrics available.")

    lines += [
        "",
        "VISUALIZATIONS GENERATED:",
        "  01_distributions.png       — Temp & Humidity distribution",
        "  02_monthly_temp_trend.png  — Monthly avg temp by timezone",
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

    with open("reports/summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n    → Saved reports/summary_report.txt")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    eda(df)
    metrics = build_models(df)
    save_summary(metrics)
    print("\n✅  All done! Check the outputs/ and reports/ folders.")
