import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("Cleaned Data")
OUTPUT_DIR = Path("Cleaned Data")
HOURS_PER_DAY = 48

def load_hourly_temps(path, label):
    print(f"Loading {label} hourly data...")
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df[['datetime_utc', 'T_avg']].copy()
    df.columns = ['datetime', 'temp']
    df['date'] = df['datetime'].dt.date
    df['slot'] = df.groupby('date').cumcount()

    days = df.groupby('date').filter(lambda g: len(g) == HOURS_PER_DAY)
    print(f"  Total hourly rows: {len(df):,}")
    print(f"  Complete days (48 slots): {days['date'].nunique():,}")

    pivoted = days.pivot(index='date', columns='slot', values='temp')
    pivoted.columns = [f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]
    return pivoted

def load_daily_full(path, label):
    print(f"Loading {label} daily data...")
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = df['date'].dt.date
    return df

def add_engineered_features(df):
    df['T_range'] = df['T_max'] - df['T_min']
    doy = pd.to_datetime(df['date']).dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['T_avg_lag1'] = df['T_avg'].shift(1)
    # Remove first row of each station after concat (lag1 will be NaN)
    return df

# --- Load targets (hourly) ---
sbd_targets = load_hourly_temps(DATA_DIR / "Sbd_nord_hourly.csv", "Sbd Nord")
vac_targets = load_hourly_temps(DATA_DIR / "Vacarisses_hourly.csv", "Vacarisses")

# --- Load features (daily) with all columns ---
sbd_feat = load_daily_full(DATA_DIR / "Sbd_nord_daily.csv", "Sbd Nord")
vac_feat = load_daily_full(DATA_DIR / "Vacarisses_daily.csv", "Vacarisses")

# --- Impute Sbd Nord wind (3.5% missing, by month) ---
wind_cols = ['avg_wind_kmh', 'wind_dir']
sbd_feat['month_tmp'] = pd.to_datetime(sbd_feat['date']).dt.month
for col in wind_cols:
    med = sbd_feat.groupby('month_tmp')[col].transform('median')
    sbd_feat[col] = sbd_feat[col].fillna(med)
sbd_feat.drop(columns='month_tmp', inplace=True)

# --- Build common-features dataset (both stations) ---
common_cols = ['T_avg', 'T_max', 'T_min', 'humidity_avg', 'rain_mm', 'pressure_avg']

def build_common(station_name, features, targets):
    df = features[['date'] + common_cols].merge(targets, on='date', how='inner')
    df['station'] = station_name
    df['is_sbd'] = int(station_name == 'Sbd Nord')
    df = add_engineered_features(df)
    before = len(df)
    df = df.dropna()
    print(f"  {station_name} (common): {len(df)} days (dropped {before - len(df)} NaN rows)")
    return df

sbd_common = build_common('Sbd Nord', sbd_feat, sbd_targets)
vac_common = build_common('Vacarisses', vac_feat, vac_targets)

df_common = pd.concat([sbd_common, vac_common], ignore_index=True)
print(f"Common dataset: {len(df_common)} days total")

common_feature_cols = (common_cols +
    ['T_range', 'T_avg_lag1', 'sin_doy', 'cos_doy', 'month', 'is_sbd'])
Xc = df_common[common_feature_cols].values
yc = df_common[[f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]].values

np.savez_compressed(
    OUTPUT_DIR / "ml_data_common.npz",
    X=Xc, y=yc,
    feature_names=np.array(common_feature_cols),
    target_names=np.array([f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]),
    years=pd.to_datetime(df_common['date']).dt.year.values,
    months=pd.to_datetime(df_common['date']).dt.month.values,
    days=pd.to_datetime(df_common['date']).dt.day.values,
    station_ids=df_common['is_sbd'].values
)
df_common[['date', 'station']].to_csv(OUTPUT_DIR / "ml_metadata_common.csv", index=False)
print(f"  -> {OUTPUT_DIR / 'ml_data_common.npz'} ({Xc.shape})")

# --- Build full-features dataset (Sbd Nord only, like Mac version) ---
full_cols = (common_cols +
    ['radiation_avg', 'avg_wind_kmh', 'wind_dir', 'max_wind_kmh'])

def build_full(features, targets):
    df = features[['date'] + full_cols].merge(targets, on='date', how='inner')
    df = add_engineered_features(df)
    before = len(df)
    df = df.dropna()
    print(f"  Sbd Nord (full): {len(df)} days (dropped {before - len(df)} NaN rows)")
    return df

df_full = build_full(sbd_feat, sbd_targets)

full_feature_cols = (full_cols +
    ['T_range', 'T_avg_lag1', 'sin_doy', 'cos_doy', 'month'])
Xf = df_full[full_feature_cols].values
yf = df_full[[f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]].values

np.savez_compressed(
    OUTPUT_DIR / "ml_data_full.npz",
    X=Xf, y=yf,
    feature_names=np.array(full_feature_cols),
    target_names=np.array([f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]),
    years=pd.to_datetime(df_full['date']).dt.year.values,
    months=pd.to_datetime(df_full['date']).dt.month.values,
    days=pd.to_datetime(df_full['date']).dt.day.values
)
df_full[['date']].to_csv(OUTPUT_DIR / "ml_metadata_full.csv", index=False)
print(f"  -> {OUTPUT_DIR / 'ml_data_full.npz'} ({Xf.shape})")

print("\nDone!")
