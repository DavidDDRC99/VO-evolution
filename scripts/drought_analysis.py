"""
Drought analysis module for VO-evolution project.
Calculates PET (Thornthwaite), SPI, SPEI, and identifies drought events.

Dependencies: pandas, numpy, scipy, spei
"""

import numpy as np
import pandas as pd
from scipy import stats
from spei import spi as compute_spi, spei as compute_spei


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_DIR, 'Cleaned Data')

STATION_FILES = {
    'Sabadell Nord': os.path.join(DATA_DIR, 'Sbd_nord_daily.csv'),
    'Sabadell Centre': os.path.join(DATA_DIR, 'Sbd_Centre_daily.csv'),
    'Vacarisses': os.path.join(DATA_DIR, 'Vacarisses_daily.csv'),
}


def load_station(name):
    """Load daily data for a station and return a clean DataFrame."""
    df = pd.read_csv(STATION_FILES[name])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[['T_avg', 'rain_mm']].dropna()
    return df


def aggregate_monthly(df):
    """
    Aggregate daily data to monthly:
      - rain_mm: sum (total precipitation per month)
      - T_avg: mean (mean temperature per month)
    """
    monthly = df.resample('MS').agg({
        'rain_mm': 'sum',
        'T_avg': 'mean',
    }).dropna()
    return monthly


# ---------------------------------------------------------------------------
# 2. POTENTIAL EVAPOTRANSPIRATION (PET) — THORNTHWAITE METHOD
# ---------------------------------------------------------------------------

# Theoretical sunshine hours per month for latitude ~41.5°N (Sabadell area)
# Source: standard astronomical tables
_SUN_HOURS_41N = [9.3, 10.3, 11.8, 13.4, 14.7, 15.3, 15.0, 13.9, 12.5, 11.0, 9.7, 9.0]


def thornthwaite_pet(monthly_temp, latitude=41.5):
    """
    Calculate Potential Evapotranspiration using the Thornthwaite (1948) method.

    Parameters
    ----------
    monthly_temp : pd.Series
        Monthly mean temperature (°C), index = DatetimeIndex (month start).
    latitude : float
        Latitude in degrees (default 41.5 for Sabadell area).

    Returns
    -------
    pd.Series
        Monthly PET in mm.

    Notes
    -----
    Formulas (Thornthwaite, 1948; U.S. Weather Bureau, 1958):

    1) Monthly heat index:
       I_i = (T_i / 5)^1.514    for T_i > 0

    2) Annual heat index:
       J = Σ I_i   (sum over 12 months)

    3) Exponent:
       α = 6.75×10⁻⁷ × J³  -  7.71×10⁻⁵ × J²  +  1.792×10⁻² × J  +  0.49239

    4) Unadjusted PET (30-day month, 12h daylight):
       PET₀ = 1.6 × (10 × T / J)^α    [mm/month]

    5) Corrected for actual daylight hours and month length:
       PET = PET₀ × (N / 12) × (days / 30)

       where N = theoretical sunshine hours for the month at given latitude.
    """
    # Ensure monthly_temp is a Series with DatetimeIndex
    if isinstance(monthly_temp, pd.DataFrame):
        raise ValueError("monthly_temp must be a Series, not a DataFrame")

    # Step 1: Monthly heat index (I_i)
    # Only for months with T > 0; negative T → I = 0
    I_monthly = (monthly_temp.clip(lower=0) / 5.0) ** 1.514

    # Step 2: Annual heat index (J) — computed per year
    # We need to compute J for each year, then map back to months
    df_temp = monthly_temp.to_frame('T')
    df_temp['I'] = I_monthly
    df_temp['year'] = df_temp.index.year

    # J = sum of I for each year
    J_yearly = df_temp.groupby('year')['I'].transform('sum')
    df_temp['J'] = J_yearly

    # Step 3: Exponent α
    J = df_temp['J']
    alpha = (6.75e-7 * J**3) - (7.71e-5 * J**2) + (1.792e-2 * J) + 0.49239

    # Step 4: Unadjusted PET₀ (for 30-day month, 12h daylight)
    T = df_temp['T'].clip(lower=0.001)  # avoid 0^α issues
    PET_0 = 16.0 * (10.0 * T / J) ** alpha

    # Step 5: Correct for actual month length and daylight hours
    n_days = df_temp.index.days_in_month
    month_idx = df_temp.index.month - 1  # 0-indexed

    # Use latitude-based sun hours if available, else interpolate
    sun_hours = np.array([_SUN_HOURS_41N[m] for m in month_idx])

    PET = PET_0 * (sun_hours / 12.0) * (n_days / 30.0)

    # Ensure PET is non-negative
    PET = PET.clip(lower=0)

    return PET


# ---------------------------------------------------------------------------
# 3. SPI AND SPEI CALCULATION
# ---------------------------------------------------------------------------

def compute_index(series, index_type='spei', timescale=6):
    """
    Compute SPI or SPEI using the spei package.

    Parameters
    ----------
    series : pd.Series
        Monthly values: precipitation for SPI, water balance (P - PET) for SPEI.
    index_type : str
        'spi' or 'spei'.
    timescale : int
        Accumulation window in months (3, 6, 12, etc.).

    Returns
    -------
    pd.Series
        Drought index values (same index as input).
    """
    if index_type == 'spi':
        result = compute_spi(series, timescale=timescale)
    elif index_type == 'spei':
        result = compute_spei(series, timescale=timescale)
    else:
        raise ValueError(f"index_type must be 'spi' or 'spei', got '{index_type}'")

    return result


def compute_drought_indices(monthly_df, timescales=[3, 6, 12]):
    """
    Compute SPI and SPEI at multiple timescales for a station.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Must contain 'rain_mm' and 'T_avg' columns with DatetimeIndex.
    timescales : list of int
        Accumulation windows in months.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns: SPEI_3, SPEI_6, SPEI_12, SPI_3, etc.
    """
    result = monthly_df.copy()

    # Water balance = Precipitation - PET (for SPEI)
    pet = thornthwaite_pet(monthly_df['T_avg'])
    result['PET'] = pet
    result['water_balance'] = monthly_df['rain_mm'] - pet

    for ts in timescales:
        # SPEI (needs water balance)
        result[f'SPEI_{ts}'] = compute_index(
            result['water_balance'], index_type='spei', timescale=ts
        )

        # SPI (needs precipitation only)
        result[f'SPI_{ts}'] = compute_index(
            monthly_df['rain_mm'], index_type='spi', timescale=ts
        )

    return result


# ---------------------------------------------------------------------------
# 4. DROUGHT EVENT DETECTION
# ---------------------------------------------------------------------------

def detect_drought_events(index_series, threshold=-1.0):
    """
    Identify individual drought events from a drought index time series.

    A drought event begins when the index drops below `threshold` and ends
    when it returns to >= 0.

    Parameters
    ----------
    index_series : pd.Series
        Drought index (e.g., SPEI-6), DatetimeIndex.
    threshold : float
        Onset threshold (default -1.0 = moderate drought).

    Returns
    -------
    pd.DataFrame
        Each row = one drought event with columns:
        - start: start date
        - end: end date
        - duration_months: length of event
        - severity: minimum index value during event
        - mean_index: average index value during event
    """
    dry = index_series < threshold
    events = []

    in_drought = False
    start = None
    values = []

    for date, is_dry in dry.items():
        if is_dry and not in_drought:
            # Drought onset
            in_drought = True
            start = date
            values = [index_series[date]]
        elif is_dry and in_drought:
            # Drought continues
            values.append(index_series[date])
        elif not is_dry and in_drought:
            # Drought ends (index returned to >= 0)
            end = date
            duration = len(values)
            events.append({
                'start': start,
                'end': end,
                'duration_months': duration,
                'severity': min(values),
                'mean_index': np.mean(values),
            })
            in_drought = False
            values = []

    # Handle ongoing drought at end of series
    if in_drought:
        events.append({
            'start': start,
            'end': index_series.index[-1],
            'duration_months': len(values),
            'severity': min(values),
            'mean_index': np.mean(values),
        })

    if not events:
        return pd.DataFrame(columns=[
            'start', 'end', 'duration_months', 'severity', 'mean_index'
        ])

    return pd.DataFrame(events)


def drought_statistics(drought_df, index_df, index_col='SPEI_6'):
    """
    Compute annual drought statistics from detected events.

    Parameters
    ----------
    drought_df : pd.DataFrame
        Output of detect_drought_events().
    index_df : pd.DataFrame
        Monthly index values (for counting total months).

    Returns
    -------
    pd.DataFrame
        Annual stats: n_events, total_months_in_drought, mean_severity,
        worst_severity, mean_duration.
    """
    if drought_df.empty:
        years = range(index_df.index.year.min(), index_df.index.year.max() + 1)
        return pd.DataFrame({
            'year': list(years),
            'n_events': 0,
            'total_months_in_drought': 0,
            'mean_severity': np.nan,
            'worst_severity': np.nan,
            'mean_duration': np.nan,
        })

    drought_df = drought_df.copy()
    drought_df['year'] = drought_df['start'].dt.year

    annual = []
    for year, group in drought_df.groupby('year'):
        annual.append({
            'year': year,
            'n_events': len(group),
            'total_months_in_drought': group['duration_months'].sum(),
            'mean_severity': group['severity'].mean(),
            'worst_severity': group['severity'].min(),
            'mean_duration': group['duration_months'].mean(),
        })

    return pd.DataFrame(annual)


# ---------------------------------------------------------------------------
# 5. TREND ANALYSIS (using existing statistical_tests.py)
# ---------------------------------------------------------------------------

def analyze_drought_trends(annual_stats, index_annual_mean):
    """
    Run Mann-Kendall trend tests on drought statistics.

    Parameters
    ----------
    annual_stats : pd.DataFrame
        From drought_statistics().
    index_annual_mean : pd.Series
        Annual mean of the drought index.

    Returns
    -------
    dict of results from statistical_tests.mk_analysis_series
    """
    import sys
    sys.path.insert(0, _SCRIPT_DIR)
    from statistical_tests import mk_analysis_series

    results = {}

    # Trend in number of events per year
    if 'n_events' in annual_stats.columns:
        results['n_events'] = mk_analysis_series(
            annual_stats['year'].values,
            annual_stats['n_events'].values,
            label='Number of drought events per year'
        )

    # Trend in total months in drought per year
    if 'total_months_in_drought' in annual_stats.columns:
        results['total_months'] = mk_analysis_series(
            annual_stats['year'].values,
            annual_stats['total_months_in_drought'].values,
            label='Months in drought per year'
        )

    # Trend in mean severity
    if 'mean_severity' in annual_stats.columns:
        valid = annual_stats.dropna(subset=['mean_severity'])
        if len(valid) >= 4:
            results['severity'] = mk_analysis_series(
                valid['year'].values,
                valid['mean_severity'].values,
                label='Mean drought severity per year'
            )

    # Trend in the index itself (annual mean)
    valid_idx = index_annual_mean.dropna()
    if len(valid_idx) >= 4:
        results['index_mean'] = mk_analysis_series(
            valid_idx.index.year.values,
            valid_idx.values,
            label='Annual mean of the drought index'
        )

    return results


# ---------------------------------------------------------------------------
# 6. CONVENIENCE: FULL ANALYSIS FOR ONE STATION
# ---------------------------------------------------------------------------

def full_analysis(station_name, timescale=6, threshold=-1.0):
    """
    Run complete drought analysis for one station.

    Returns
    -------
    dict with keys:
        'station': station name
        'monthly': monthly DataFrame with indices
        'drought_events': detected events
        'annual_stats': annual drought statistics
        'trends': Mann-Kendall trend results
        'pet': PET series
    """
    print(f"\n{'='*60}")
    print(f"  DROUGHT ANALYSIS: {station_name}")
    print(f"{'='*60}")

    # Load and aggregate
    print(f"1. Loading daily data...")
    daily = load_station(station_name)
    print(f"   Period: {daily.index[0].date()} to {daily.index[-1].date()} ({len(daily)} days)")

    print(f"2. Aggregating to monthly...")
    monthly = aggregate_monthly(daily)
    print(f"   {len(monthly)} months")

    # Compute PET
    print(f"3. Computing PET (Thornthwaite)...")
    pet = thornthwaite_pet(monthly['T_avg'])
    monthly['PET'] = pet

    # Compute indices
    print(f"4. Computing SPI and SPEI (timescale {timescale} months)...")
    monthly = compute_drought_indices(monthly, timescales=[timescale])

    # Detect events
    index_col = f'SPEI_{timescale}'
    print(f"5. Detecting drought events ({index_col} < {threshold})...")
    drought_events = detect_drought_events(monthly[index_col], threshold=threshold)
    print(f"   {len(drought_events)} events detected")

    # Annual stats
    print(f"6. Computing annual statistics...")
    annual = drought_statistics(drought_events, monthly, index_col)

    # Trends
    print(f"7. Analyzing trends (Mann-Kendall)...")
    annual_mean_index = monthly[index_col].resample('YS').mean()
    trends = analyze_drought_trends(annual, annual_mean_index)

    print(f"\n RESULTS:")
    for key, result in trends.items():
        print(f"\n{result['summary']}")

    return {
        'station': station_name,
        'monthly': monthly,
        'drought_events': drought_events,
        'annual_stats': annual,
        'trends': trends,
        'pet': pet,
    }


# ---------------------------------------------------------------------------
# MAIN (for testing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for station in STATION_FILES:
        result = full_analysis(station, timescale=6, threshold=-1.0)
        print(f"\nDrought events ({station}):")
        if not result['drought_events'].empty:
            print(result['drought_events'].to_string(index=False))
        else:
            print("  No drought events detected.")
