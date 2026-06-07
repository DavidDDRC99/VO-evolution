import pandas as pd
import numpy as np

def seasonal_sens(df):
    monthly = df.groupby(['year', 'month'])['rain_mm'].sum()
    slopes = []
    for m in range(1, 13):
        try:
            vals = monthly.xs(m, level='month').values
            years = monthly.xs(m, level='month').index.values
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    slopes.append((vals[j] - vals[i]) / (years[j] - years[i]))
        except:
            pass
    slopes = np.array(slopes)
    return np.median(slopes), np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)

urls = {
    'Sabadell Centre': 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Sbd_Centre_daily.csv',
    'Sabadell Nord': 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Sbd_nord_daily.csv',
    'Vacarisses': 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Vacarisses_daily.csv',
}

for name, url in urls.items():
    df = pd.read_csv(url)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    slope, lo, hi = seasonal_sens(df)
    p = f"{df['year'].min()}-{df['year'].max()}"
    print(f"{name} ({p})")
    print(f"  Seasonal Sen's slope: {slope:.2f} mm/any")
    print(f"  IC 95%: [{lo:.2f}, {hi:.2f}]")
    print()
