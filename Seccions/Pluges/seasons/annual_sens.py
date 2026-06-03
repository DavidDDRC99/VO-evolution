import pandas as pd
import numpy as np

def annual_sens(df):
    annual = df.groupby('year')['rain_mm'].sum()
    years = annual.index.values
    vals = annual.values
    slopes = []
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            slopes.append((vals[j] - vals[i]) / (years[j] - years[i]))
    slopes = np.array(slopes)
    return np.median(slopes)

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
    slope = annual_sens(df)
    print(f"{name}: {slope:.1f} mm/any")
