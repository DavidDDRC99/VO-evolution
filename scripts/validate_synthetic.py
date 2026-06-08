import pandas as pd
import numpy as np

df = pd.read_csv('Cleaned Data/Sbd_Centre_hourly_synthetic.csv', parse_dates=['datetime'])
hourly = pd.read_csv('Cleaned Data/Sbd_nord_hourly.csv', parse_dates=['datetime_utc'])
daily = pd.read_csv('Cleaned Data/Sbd_Centre_daily.csv', index_col=0, parse_dates=['date'])
daily = daily.set_index('date')

print('=== SYNTHETIC DATA QUALITY CHECKS ===\n')

df['date'] = df['datetime'].dt.date
expected_days = len(daily)
actual_days = df['date'].nunique()
print(f'1. Completeness: {actual_days}/{expected_days} days ({actual_days/expected_days:.1%})')

slots = df.groupby('date').size()
print(f'2. Slots per day: {slots.min()}-{slots.max()} (expected 48)')

diffs = df['datetime'].diff().dropna()
gap_max = diffs.max()
print(f'3. Max timestep gap: {gap_max}')
print(f'   Non-30min gaps: {(diffs != pd.Timedelta(minutes=30)).sum()}')

print(f'4. Temp range: {df["T_30min_C"].min():.1f}C to {df["T_30min_C"].max():.1f}C')
print(f'   Mean: {df["T_30min_C"].mean():.2f}C')
print(f'   Nord mean: {hourly["T_avg"].mean():.2f}C')

daily_min = df.groupby('date')['T_30min_C'].min()
daily_max = df.groupby('date')['T_30min_C'].max()
amplitude = daily_max - daily_min
print(f'5. Diurnal amplitude: {amplitude.median():.1f}C median '
      f'({amplitude.quantile(0.05):.1f}C-{amplitude.quantile(0.95):.1f}C 90% range)')

daily_index = daily.index
violations = 0
for dt, grp in df.groupby('date'):
    dt_ts = pd.Timestamp(dt)
    if dt_ts in daily_index:
        tmin = daily.loc[dt_ts, 'T_min']
        tmax = daily.loc[dt_ts, 'T_max']
        v = ((grp['T_30min_C'] < tmin - 0.01) | (grp['T_30min_C'] > tmax + 0.01)).sum()
        violations += v
print(f'6. Bounds violations: {violations} (0 expected)')

df['month'] = df['datetime'].dt.month
monthly_centre = df.groupby('month')['T_30min_C'].mean()
hourly['month'] = hourly['datetime_utc'].dt.month
monthly_nord = hourly.groupby('month')['T_avg'].mean()
diff = (monthly_centre - monthly_nord).abs()
print(f'7. Monthly mean abs diff Nord vs Centre: {diff.mean():.2f}C avg')
print(f'   Largest monthly diff: {diff.max():.2f}C (month {diff.idxmax()})')
for m in range(1, 13):
    print(f'   Month {m:2d}: Nord={monthly_nord[m]:.1f}C  Centre={monthly_centre[m]:.1f}C  diff={diff[m]:.1f}C')
