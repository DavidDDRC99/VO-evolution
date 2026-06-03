import pandas as pd
import numpy as np

BASE = 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/'

URLS = {
    'nord_hourly': BASE + 'Sbd_nord_hourly.csv',
    'nord_daily': BASE + 'Sbd_nord_daily.csv',
    'centre_daily': BASE + 'Sbd_Centre_daily.csv',
    'vac_hourly': BASE + 'Vacarisses_hourly.csv',
    'vac_daily': BASE + 'Vacarisses_daily.csv',
}


def _drop_unnamed(df):
    for col in df.columns:
        if 'Unnamed' in str(col):
            df = df.drop(columns=[col])
    return df


def _parse_date_col(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df


def _extract_components(df, date_col='datetime_utc'):
    if date_col in df.columns:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['hour'] = df[date_col].dt.hour
        df['minute'] = df[date_col].dt.minute
        df['date'] = df[date_col].dt.normalize()
    return df


def _rename_columns(df, station='nord'):
    rename_map = {
        'humidity (%)': 'humidity',
        'rain_mm': 'rain',
        'pressure (hPa)': 'pressure',
        'radiation (W/m²)': 'radiation',
        'avg_wind_kmh': 'wind_avg',
        'max_wind_kmh': 'wind_max',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


def load_nord_hourly():
    df = pd.read_csv(URLS['nord_hourly'])
    df = _drop_unnamed(df)
    df = _parse_date_col(df, 'datetime_utc')
    df = _extract_components(df, 'datetime_utc')
    df = _rename_columns(df, 'nord')
    return df


def load_nord_daily():
    df = pd.read_csv(URLS['nord_daily'])
    df = _drop_unnamed(df)
    df = _parse_date_col(df, 'date')
    df = _extract_components(df, 'date')
    df = _rename_columns(df, 'nord')
    return df


def load_centre_daily():
    df = pd.read_csv(URLS['centre_daily'])
    df = _drop_unnamed(df)
    df = _parse_date_col(df, 'date')
    df = _extract_components(df, 'date')
    df = _rename_columns(df, 'centre')
    return df


def load_vac_hourly():
    df = pd.read_csv(URLS['vac_hourly'])
    df = _drop_unnamed(df)
    df = _parse_date_col(df, 'datetime_utc')
    df = _extract_components(df, 'datetime_utc')
    df = _rename_columns(df, 'vac')
    return df


def load_vac_daily():
    df = pd.read_csv(URLS['vac_daily'])
    df = _drop_unnamed(df)
    df = _parse_date_col(df, 'date')
    df = _extract_components(df, 'date')
    df = _rename_columns(df, 'vac')
    return df


def load_all_data():
    return {
        'nord_hourly': load_nord_hourly(),
        'nord_daily': load_nord_daily(),
        'centre_daily': load_centre_daily(),
        'vac_hourly': load_vac_hourly(),
        'vac_daily': load_vac_daily(),
    }
