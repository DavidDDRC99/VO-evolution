import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("Cleaned Data")
OUTPUT_DIR = Path("Cleaned Data")
VALIDATION_DIR = OUTPUT_DIR / "validation"
HOURS_PER_DAY = 48
SLOT_LABELS = [f'{h:02d}:{m:02d}' for h in range(24) for m in (0, 30)]

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Load hourly temperature targets from Sbd Nord ---
print("=" * 60)
print("1. Loading Sbd Nord hourly data...")
hourly = pd.read_csv(DATA_DIR / "Sbd_nord_hourly.csv", parse_dates=['datetime_utc'])
hourly = hourly[['datetime_utc', 'T_avg']].copy()
hourly.columns = ['datetime', 'temp']
hourly['date'] = hourly['datetime'].dt.date

# Keep only complete days (48 slots)
hourly['slot'] = hourly.groupby('date').cumcount()
complete = hourly.groupby('date').filter(lambda g: len(g) == HOURS_PER_DAY)
pivoted = complete.pivot(index='date', columns='slot', values='temp')
pivoted.columns = [f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]
y_targets = pivoted
print(f"  Complete days: {len(y_targets):,}")

# --- 2. Load Sbd Nord daily features ---
print("\n2. Loading Sbd Nord daily features...")
nord_daily = pd.read_csv(DATA_DIR / "Sbd_nord_daily.csv", parse_dates=['date'])
nord_daily['date'] = nord_daily['date'].dt.date

# Merge features with targets
df_nord = nord_daily.merge(y_targets, on='date', how='inner')
print(f"  Merged rows: {len(df_nord):,}")

# --- 3. Engineer features ---
print("\n3. Engineering features...")
df_nord = df_nord.sort_values('date').reset_index(drop=True)
df_nord['T_range'] = df_nord['T_max'] - df_nord['T_min']

doy = pd.to_datetime(df_nord['date']).dt.dayofyear
df_nord['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
df_nord['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
df_nord['month'] = pd.to_datetime(df_nord['date']).dt.month
df_nord['T_avg_lag1'] = df_nord['T_avg'].shift(1)

# Features available in Sbd Centre (no humidity, pressure, radiation)
centre_features = [
    'T_avg', 'T_max', 'T_min', 'rain_mm',
    'avg_wind_kmh', 'wind_dir', 'max_wind_kmh',
    'T_range', 'T_avg_lag1', 'sin_doy', 'cos_doy', 'month'
]

# Impute missing wind data in Sbd Nord (3.5% missing, early years)
for col in ['avg_wind_kmh', 'wind_dir']:
    med = df_nord.groupby('month')[col].transform('median')
    df_nord[col] = df_nord[col].fillna(med)

# Drop any remaining NaN rows (e.g., first row from T_avg_lag1)
before = len(df_nord)
df_nord = df_nord.dropna(subset=centre_features)
print(f"  Dropped {before - len(df_nord)} rows with NaNs, {len(df_nord)} remaining")

# --- 4. Train MLP on Sbd Nord ---
print("\n4. Training MLP on Sbd Nord (Centre-compatible features)...")
target_cols = [f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]
X = df_nord[centre_features].values
y = df_nord[target_cols].values

# Temporal split (80/20)
n = len(X)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# MLP with known best hyperparams from common-features model
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    learning_rate_init=0.01,
    alpha=0.001,
    max_iter=500,
    random_state=42,
    verbose=False
)

print("  Fitting MLP...")
mlp.fit(X_train_s, y_train)
print(f"  Converged in {mlp.n_iter_} iterations")

# Evaluate on held-out Sbd Nord test set
y_pred = mlp.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
bias = np.mean(y_pred - y_test)
print(f"\n  Sbd Nord Test Set Results:")
print(f"    RMSE: {rmse:.3f}C")
print(f"    R²:   {r2:.4f}")
print(f"    Bias: {bias:.4f}C")

# Per-slot RMSE
slot_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))
print(f"    RMSE/slot: {slot_rmse.mean():.3f} +/- {slot_rmse.std():.3f}C")

# --- 5. Generate Sbd Centre synthetic hourly data ---
print("\n" + "=" * 60)
print("5. Generating Sbd Centre synthetic hourly data...")
centre_daily = pd.read_csv(DATA_DIR / "Sbd_Centre_daily.csv", parse_dates=['date'], index_col=0)
centre_daily['date'] = centre_daily['date'].dt.date

# Engineer features
centre = centre_daily.sort_values('date').reset_index(drop=True)
centre['T_range'] = centre['T_max'] - centre['T_min']

doy_centre = pd.to_datetime(centre['date']).dt.dayofyear
centre['sin_doy'] = np.sin(2 * np.pi * doy_centre / 365.25)
centre['cos_doy'] = np.cos(2 * np.pi * doy_centre / 365.25)
centre['month'] = pd.to_datetime(centre['date']).dt.month
centre['T_avg_lag1'] = centre['T_avg'].shift(1)

# Drop first row (NaN lag1)
before = len(centre)
centre = centre.dropna(subset=centre_features).reset_index(drop=True)
print(f"  Sbd Centre days with complete features: {len(centre):,} (dropped {before - len(centre)})")

# Predict
X_centre = scaler.transform(centre[centre_features].values)
y_centre_pred = mlp.predict(X_centre)

# --- 6. Post-process: enforce daily min/max constraints ---
print("\n6. Post-processing: enforcing daily min/max constraints...")
enforced = y_centre_pred.copy()
for i in range(len(centre)):
    T_min = centre.iloc[i]['T_min']
    T_max = centre.iloc[i]['T_max']
    enforced[i] = np.clip(enforced[i], T_min, T_max)

# --- 7. Build long-format CSV ---
print("\n7. Building output CSV...")
rows = []
for i, row in centre.iterrows():
    day_date = row['date']
    day_start = pd.Timestamp(day_date)
    for slot in range(HOURS_PER_DAY):
        dt = day_start + pd.Timedelta(minutes=slot * 30)
        rows.append({
            'datetime': dt,
            'T_30min_C': enforced[i, slot]
        })

df_out = pd.DataFrame(rows)
csv_path = OUTPUT_DIR / "Sbd_Centre_hourly_synthetic.csv"
df_out.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")
print(f"  Rows: {len(df_out):,}")
print(f"  Date range: {df_out['datetime'].min()} to {df_out['datetime'].max()}")

# Quick sanity checks
print(f"\n  Temperature range: {df_out['T_30min_C'].min():.1f}C to {df_out['T_30min_C'].max():.1f}C")
print(f"  Mean temperature:  {df_out['T_30min_C'].mean():.2f}C")

# --- 8. Validation plots ---
print("\n" + "=" * 60)
print("8. Generating validation figures...")
plt.rcParams.update({'font.size': 10})

# 8a. Seasonal cycle: monthly means
print("  8a. Seasonal cycle comparison...")
centre_df = df_out.copy()
centre_df['date'] = centre_df['datetime'].dt.date
centre_daily_mean = centre_df.groupby('date')['T_30min_C'].mean().reset_index()
centre_daily_mean['month'] = pd.to_datetime(centre_daily_mean['date']).dt.month
monthly_centre = centre_daily_mean.groupby('month')['T_30min_C'].mean()

# Nord monthly means from daily data
nord_daily_plot = nord_daily.copy()
nord_daily_plot['month'] = pd.to_datetime(nord_daily_plot['date']).dt.month
monthly_nord = nord_daily_plot.groupby('month')['T_avg'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
months = range(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.plot(months, monthly_nord.values, 'o-', color='#2196F3', linewidth=2, label='Sbd Nord (real)')
ax.plot(months, monthly_centre.values, 's--', color='#FF5722', linewidth=2, label='Sbd Centre (synthetic)')
ax.set_xlabel('Month')
ax.set_ylabel('Mean Temperature (°C)')
ax.set_title('Seasonal Cycle Comparison')
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(VALIDATION_DIR / "sbd_centre_vs_nord_seasonal_cycle.png", dpi=150)
plt.close(fig)
print("    Saved: sbd_centre_vs_nord_seasonal_cycle.png")

# 8b. Average daily curve per season
print("  8b. Daily curve shapes per season...")
seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11]}

# Nord daily curves from test-set predictions
nord_pred_df = pd.DataFrame(y_pred, index=df_nord.iloc[split:].index, columns=target_cols)
nord_pred_df['date'] = df_nord.iloc[split:]['date'].values
nord_pred_df['month'] = pd.to_datetime(nord_pred_df['date']).dt.month

# Centre daily curves
centre_curve_df = pd.DataFrame(enforced, columns=target_cols)
centre_curve_df['date'] = centre['date'].values
centre_curve_df['month'] = pd.to_datetime(centre_curve_df['date']).dt.month

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
slot_hours = np.arange(0, 24, 0.5)

for idx, (season_name, season_months) in enumerate(seasons.items()):
    ax = axes[idx]

    # Nord predicted (test set)
    nord_season = nord_pred_df[nord_pred_df['month'].isin(season_months)]
    if len(nord_season) > 0:
        nord_mean = nord_season[target_cols].mean().values
        nord_std = nord_season[target_cols].std().values
        ax.plot(slot_hours, nord_mean, color='#2196F3', linewidth=2, label='Sbd Nord (predicted)')
        ax.fill_between(slot_hours, nord_mean - nord_std, nord_mean + nord_std,
                        color='#2196F3', alpha=0.15)

    # Centre synthetic
    centre_season = centre_curve_df[centre_curve_df['month'].isin(season_months)]
    if len(centre_season) > 0:
        centre_mean = centre_season[target_cols].mean().values
        centre_std = centre_season[target_cols].std().values
        ax.plot(slot_hours, centre_mean, '--', color='#FF5722', linewidth=2, label='Sbd Centre (synthetic)')
        ax.fill_between(slot_hours, centre_mean - centre_std, centre_mean + centre_std,
                        color='#FF5722', alpha=0.15)

    ax.set_title(season_name)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(VALIDATION_DIR / "sbd_centre_vs_nord_daily_curves.png", dpi=150)
plt.close(fig)
print("    Saved: sbd_centre_vs_nord_daily_curves.png")

# 8c. Extreme values comparison: daily T_min/T_max
print("  8c. Extreme values comparison...")
centre_df['date'] = centre_df['datetime'].dt.date
centre_daily_min = centre_df.groupby('date')['T_30min_C'].min().reset_index(name='pred_min')
centre_daily_max = centre_df.groupby('date')['T_30min_C'].max().reset_index(name='pred_max')
centre_daily_stats = centre[['date', 'T_min', 'T_max']].merge(
    centre_daily_min, on='date').merge(centre_daily_max, on='date')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(centre_daily_stats['T_min'], centre_daily_stats['pred_min'],
           alpha=0.3, s=5, color='#2196F3')
lims = [min(centre_daily_stats['T_min'].min(), centre_daily_stats['pred_min'].min()),
        max(centre_daily_stats['T_max'].max(), centre_daily_stats['pred_max'].max())]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Observed T_min (°C)')
ax.set_ylabel('Synthetic T_min (°C)')
ax.set_title(f'T_min (enforced: clip)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(centre_daily_stats['T_max'], centre_daily_stats['pred_max'],
           alpha=0.3, s=5, color='#FF5722')
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Observed T_max (°C)')
ax.set_ylabel('Synthetic T_max (°C)')
ax.set_title('T_max (enforced: clip)')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(VALIDATION_DIR / "sbd_centre_vs_nord_extremes.png", dpi=150)
plt.close(fig)
print("    Saved: sbd_centre_vs_nord_extremes.png")

# 8d. Prediction error distribution on Sbd Nord test set
print("  8d. Error distribution on Sbd Nord test set...")
errors = (y_pred - y_test).ravel()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors, bins=80, color='#4CAF50', alpha=0.7, edgecolor='none')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_xlabel('Prediction Error (°C)')
ax.set_ylabel('Frequency')
ax.set_title(f'Sbd Nord Test Set Error Distribution (RMSE={rmse:.3f}°C)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(VALIDATION_DIR / "sbd_nord_heldout_errors.png", dpi=150)
plt.close(fig)
print("    Saved: sbd_nord_heldout_errors.png")

print("\n" + "=" * 60)
print("Done! Synthetic hourly data generated successfully.")
print(f"  Output: {csv_path}")
print(f"  Figures: {VALIDATION_DIR}/")
