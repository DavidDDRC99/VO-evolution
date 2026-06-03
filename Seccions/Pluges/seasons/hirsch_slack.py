import pandas as pd
import numpy as np

def seasonal_sens_hirsch_slack(df):
    """Hirsch-Slack Seasonal Sen's slope: per-month Sen's slope, then median."""
    monthly = df.groupby(['year', 'month'])['rain_mm'].sum()
    month_slopes = []
    for m in range(1, 13):
        try:
            vals = monthly.xs(m, level='month').values
            years = monthly.xs(m, level='month').index.values
            slopes = []
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    slopes.append((vals[j] - vals[i]) / (years[j] - years[i]))
            month_slopes.append(np.median(slopes))
        except:
            pass
    return np.median(month_slopes)

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
    slope = seasonal_sens_hirsch_slack(df)
    print(f"{name}: {slope:.2f} mm/any")
