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

def load_daily(path, label):
    print(f"Loading {label} daily data...")
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = df['date'].dt.date
    return df[['date', 'T_min', 'T_max', 'T_avg']].copy()

sbd_targets = load_hourly_temps(DATA_DIR / "Sbd_nord_hourly.csv", "Sbd Nord")
vac_targets = load_hourly_temps(DATA_DIR / "Vacarisses_hourly.csv", "Vacarisses")

sbd_features = load_daily(DATA_DIR / "Sbd_nord_daily.csv", "Sbd Nord")
vac_features = load_daily(DATA_DIR / "Vacarisses_daily.csv", "Vacarisses")

def build_dataset(targets, features, station_name):
    merged = features.merge(targets, on='date', how='inner')
    merged['station'] = station_name
    print(f"  {station_name}: {len(merged)} days with complete feature+target data")
    return merged

sbd = build_dataset(sbd_targets, sbd_features, 'Sbd Nord')
vac = build_dataset(vac_targets, vac_features, 'Vacarisses')

df = pd.concat([sbd, vac], ignore_index=True)
print(f"\nCombined dataset: {len(df)} days, {len(df.columns)} columns")

# Seasonal features
df['date_obj'] = pd.to_datetime(df['date'])
day_of_year = df['date_obj'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['cos_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['month'] = df['date_obj'].dt.month
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
df['is_sbd'] = (df['station'] == 'Sbd Nord').astype(int)

# Drop rows with missing targets or features
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows with NaN (T_min/T_max missing)")

# Separate features and targets
feature_cols = ['T_min', 'T_max', 'T_avg', 'sin_day', 'cos_day',
                'sin_month', 'cos_month', 'is_sbd']
target_cols = [f'temp_{s:02d}' for s in range(HOURS_PER_DAY)]
meta_cols = ['date', 'station', 'date_obj']

X = df[feature_cols].values
y = df[target_cols].values
meta = df[meta_cols]

print(f"\nFeature matrix X: {X.shape}")
print(f"Target matrix y: {y.shape}")
print(f"Features: {feature_cols}")
print(f"Target slots: {target_cols[0]} .. {target_cols[-1]}")

np.savez_compressed(
    OUTPUT_DIR / "ml_data.npz",
    X=X, y=y,
    feature_names=np.array(feature_cols),
    target_names=np.array(target_cols),
    years=df['date_obj'].dt.year.values,
    months=df['date_obj'].dt.month.values,
    days=df['date_obj'].dt.day.values,
    station_ids=df['is_sbd'].values
)
meta.to_csv(OUTPUT_DIR / "ml_metadata.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'ml_data.npz'} + ml_metadata.csv")
print("Done!")
