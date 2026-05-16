"""
Inject statistical test cells into the analysis notebooks.
Run: python scripts/inject_stat_tests.py
"""

import nbformat as nbf
import os

PROJECT = r"C:\Users\David\Documents\VO project\VO-evolution"
SCRIPTS = os.path.join(PROJECT, "scripts")


def new_code(source):
    return nbf.v4.new_code_cell(source)


def new_md(source):
    return nbf.v4.new_markdown_cell(source)


# --- Generic imports cell (shared across notebooks) ---
IMPORTS_CODE = """\
# === Statistical trend analysis ===
import sys
sys.path.insert(0, r"{{SCRIPTS}}")
from statistical_tests import (
    mk_analysis_series, mann_kendall, sens_slope, pettitt_test,
    seasonal_mann_kendall, print_mk_table
)
from scipy import stats
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
"""
IMPORTS_CELL = new_code(IMPORTS_CODE)

# =============================================================================
# RAIN_SEASONS
# =============================================================================
RAIN_CELLS = [
    IMPORTS_CELL,
    new_md(
        "## 6. Trend Analysis — Statistical Tests\n\n"
        "Mann-Kendall trend test, Sen's slope, and Pettitt change-point detection "
        "on monthly and annual rainfall series."
    ),
    new_md("### 6.1 Seasonal Mann-Kendall (accounting for monthly seasonality)"),
    new_code(
        """\
# Seasonal Mann-Kendall on monthly rainfall
# This accounts for the seasonal cycle and tests for a monotonic trend
results_smk = []
for name, df in [('Sbd Centre', df_sbd_centre), ('Sbd Nord', df_sbd_nord), ('Vacarisses', df_vac)]:
    r = seasonal_mann_kendall(df, 'rain_mm', month_col='month', year_col='year')
    r['station'] = name
    results_smk.append(r)
    print(f"\\n{'='*60}")
    print(f"  Seasonal Mann-Kendall — {name}")
    print(f"{'='*60}")
    print(f"  S_total = {r['S_total']:.1f}")
    print(f"  Z = {r['Z']:.4f}")
    print(f"  p-value = {r['p_value']:.4f}")
    print(f"  Trend: {r['trend']}")
    print(f"  Significant at alpha=0.05: {'YES' if r['p_value'] < 0.05 else 'NO'}")
    # Per-month detail
    print(f"\\n  Per-month MK statistics:")
    for mr in r['monthly_results']:
        print(f"    Month {mr['month']:2d}: S = {mr['S']:+6.1f}, n = {mr['n']}")
"""
    ),
    new_md("### 6.2 Mann-Kendall per Month (individual months)"),
    new_code(
        """\
# MK + Sen's slope for each month separately
month_names_en = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                  7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

for name, df in [('Sbd Centre', df_sbd_centre), ('Sbd Nord', df_sbd_nord), ('Vacarisses', df_vac)]:
    monthly = df.groupby(['year', 'month'])['rain_mm'].sum().reset_index()
    print(f"\\n{'='*60}")
    print(f"  MK per Month — {name}")
    print(f"{'='*60}")
    month_results = []
    for m in range(1, 13):
        sub = monthly[monthly['month'] == m].sort_values('year')
        years = sub['year'].values
        vals = sub['rain_mm'].values
        r = mk_analysis_series(years, vals, f'  {month_names_en[m]}')
        month_results.append(r)
    print_mk_table(month_results)
"""
    ),
    new_md("### 6.3 Annual Rainfall Trend"),
    new_code(
        """\
# Annual rainfall totals
for name, df in [('Sbd Centre', df_sbd_centre), ('Sbd Nord', df_sbd_nord), ('Vacarisses', df_vac)]:
    annual = df.groupby('year')['rain_mm'].sum().reset_index()
    r = mk_analysis_series(annual['year'].values, annual['rain_mm'].values, f'Annual rainfall — {name}')
    print(f"\\n{r['summary']}")
"""
    ),
    new_md("### 6.4 Pettitt Test on Annual Series"),
    new_code(
        """\
# Pettitt change-point on annual totals
print("\\nPettitt change-point detection on annual rainfall:")
for name, df in [('Sbd Centre', df_sbd_centre), ('Sbd Nord', df_sbd_nord), ('Vacarisses', df_vac)]:
    annual = df.groupby('year')['rain_mm'].sum().reset_index()
    pt = pettitt_test(annual['rain_mm'].values)
    cp_year = annual['year'].values[int(pt['cp_index'])] if not np.isnan(pt['cp_index']) else 'N/A'
    print(f"  {name:15s}: CP at year {cp_year}, U={pt['U_stat']:.1f}, "
          f"p={pt['p_value']:.4f} {'*' if pt['significant_0.05'] else ''}")
"""
    ),
]

# =============================================================================
# HOURLY_RAIN
# =============================================================================
HOURLY_CELLS = [
    IMPORTS_CELL,
    new_md(
        "## 8. Trend Analysis — Statistical Tests\n\n"
        "Testing for trends in hourly rainfall patterns using Mann-Kendall "
        "and comparing distributions between periods."
    ),
    new_md("### 8.1 Trend in Mean Hourly Rainfall by Year"),
    new_code(
        """\
# MK on annual mean hourly rainfall
for name, df, valid_years in [
    ('Vacarisses', df_vac_h, valid_years_vac),
    ('Sbd Nord', df_sbd_h, valid_years_sbd)
]:
    df_v = df[df['year'].isin(valid_years)]
    yearly_mean = df_v.groupby('year')['rain_mm'].mean().reset_index()
    r = mk_analysis_series(
        yearly_mean['year'].values, yearly_mean['rain_mm'].values,
        f'Mean hourly rain — {name}'
    )
    print(f"\\n{r['summary']}")
"""
    ),
    new_md("### 8.2 KS Test: Comparing Rain Distributions (Early vs Recent)"),
    new_code(
        """\
# Kolmogorov-Smirnov test comparing rainfall distributions
# between the first and last 5 years of each station
for name, df, valid_years in [
    ('Vacarisses', df_vac_h, valid_years_vac),
    ('Sbd Nord', df_sbd_h, valid_years_sbd)
]:
    vy = sorted(valid_years)
    early = df[df['year'].isin(vy[:5])]['rain_mm'].dropna().values
    late = df[df['year'].isin(vy[-5:])]['rain_mm'].dropna().values
    if len(early) < 10 or len(late) < 10:
        print(f"\\n  {name}: insufficient data for KS test")
        continue
    ks_stat, ks_p = stats.ks_2samp(early, late)
    print(f"\\n  {name}: KS test (first 5 vs last 5 years)")
    print(f"    D = {ks_stat:.4f}, p = {ks_p:.4f}")
    print(f"    {'Different distributions (p<0.05)' if ks_p < 0.05 else 'No significant difference'}")
    # Also compare means
    t_stat, t_p = stats.ttest_ind(early, late)
    print(f"    T-test: t = {t_stat:.4f}, p = {t_p:.4f}")
    print(f"    Early mean = {early.mean():.4f}, Late mean = {late.mean():.4f}")
"""
    ),
    new_md("### 8.3 Seasonal Mann-Kendall on Monthly Mean Hourly Rain"),
    new_code(
        """\
# Seasonal MK on monthly mean hourly rainfall
for name, df, valid_years in [
    ('Vacarisses', df_vac_h, valid_years_vac),
    ('Sbd Nord', df_sbd_h, valid_years_sbd)
]:
    df_v = df[df['year'].isin(valid_years)].copy()
    df_v['month'] = pd.to_datetime(df_v['datetime_utc']).dt.month
    monthly = df_v.groupby(['year', 'month'])['rain_mm'].mean().reset_index()
    r = seasonal_mann_kendall(monthly, 'rain_mm', 'month', 'year')
    print(f"\\n{'='*50}")
    print(f"  Seasonal MK — {name}")
    print(f"  Z = {r['Z']:.4f}, p = {r['p_value']:.4f}")
    print(f"  Trend: {r['trend']}")
"""
    ),
]

# =============================================================================
# INTENSITY
# =============================================================================
INTENSITY_CELLS = [
    IMPORTS_CELL,
    new_md(
        "## 7. Trend Analysis — Statistical Tests\n\n"
        "Mann-Kendall tests on rainfall intensity metrics."
    ),
    new_md("### 7.1 Trend in Maximum Intensity by Year"),
    new_code(
        """\
# MK on maximum 30-min, hourly, and daily rainfall per year
# (requires the max_df from section 6)
try:
    print("MK on maximum intensity per year:")
    for col, label in [('max_30min', 'Max 30-min (Sbd Nord)'),
                       ('max_hourly', 'Max hourly (Sbd Nord)'),
                       ('max_daily', 'Max daily (Sbd Nord)'),
                       ('max_daily_vac', 'Max daily (Vacarisses)'),
                       ('max_daily_centre', 'Max daily (Sbd Centre)')]:
        sub = max_df[['year', col]].dropna()
        if len(sub) < 4:
            continue
        r = mk_analysis_series(sub['year'].values, sub[col].values, label)
        print(f"\\n{r['summary']}")
except NameError:
    print("Run section 6 first to compute max_df, or load data directly:")
    print("# For a quick test, use the daily data:")
    # Fallback: use daily data
    for name, url in [('Sbd Nord', 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Sbd_nord_daily.csv'),
                      ('Vacarisses', 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Vacarisses_daily.csv'),
                      ('Sbd Centre', 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Sbd_Centre_daily.csv')]:
        df = pd.read_csv(url)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df['year'] = pd.to_datetime(df['date']).dt.year
        annual_max = df.groupby('year')['rain_mm'].max().reset_index()
        r = mk_analysis_series(annual_max['year'].values, annual_max['rain_mm'].values,
                               f'Max daily — {name}')
        print(f"\\n{r['summary']}")
"""
    ),
    new_md("### 7.2 Trend in Frequency of Heavy Rainfall Events"),
    new_code(
        """\
# MK on count of events above percentiles
for name, url in [('Sbd Nord', 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Sbd_nord_hourly.csv'),
                  ('Vacarisses', 'https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/refs/heads/main/Cleaned%20Data/Vacarisses_hourly.csv')]:
    df = pd.read_csv(url)
    df['year'] = pd.to_datetime(df['datetime_utc']).dt.year
    yearly_counts = []
    for yr in sorted(df['year'].unique()):
        sub = df[(df['year'] == yr) & (df['rain_mm'] > 0)]
        if len(sub) < 10:
            continue
        p95 = sub['rain_mm'].quantile(0.95)
        count = (sub['rain_mm'] > p95).sum()
        yearly_counts.append({'year': yr, 'count': count, 'p95': p95})
    counts_df = pd.DataFrame(yearly_counts)
    r = mk_analysis_series(
        counts_df['year'].values, counts_df['count'].values,
        f'Heavy events (>P95) per year — {name}'
    )
    print(f"\\n{r['summary']}")
"""
    ),
]

# =============================================================================
# TEMPERATURES_NIGHT
# =============================================================================
TEMP_CELLS = [
    IMPORTS_CELL,
    new_md(
        "## 14. Trend Analysis — Statistical Tests\n\n"
        "Mann-Kendall trend tests on nighttime summer temperature metrics."
    ),
    new_md("### 14.1 Trend in Nighttime T_min (mean per year)"),
    new_code(
        """\
# MK on mean T_min per year (summer nights, Sbd Nord)
nightly_mean = nights_valid.groupby('year')['night_T_min'].mean().reset_index()
r_nord = mk_analysis_series(
    nightly_mean['year'].values, nightly_mean['night_T_min'].values,
    'Mean night T_min — Sbd Nord'
)
print(f"\\n{r_nord['summary']}")

# Sbd Centre daily T_min for comparison
daily_mean = daily_centre_valid.groupby('year')['T_min'].mean().reset_index()
r_centre = mk_analysis_series(
    daily_mean['year'].values, daily_mean['T_min'].values,
    'Mean T_min — Sbd Centre'
)
print(f"\\n{r_centre['summary']}")
"""
    ),
    new_md("### 14.2 Trend in Nighttime T_max (mean per year)"),
    new_code(
        """\
nightly_tmax = nights_valid.groupby('year')['night_T_max'].mean().reset_index()
r = mk_analysis_series(
    nightly_tmax['year'].values, nightly_tmax['night_T_max'].values,
    'Mean night T_max — Sbd Nord'
)
print(f"\\n{r['summary']}")
"""
    ),
    new_md("### 14.3 Trend in Warm Nights per Threshold"),
    new_code(
        """\
# MK on count of warm nights for each threshold
print("\\nMK on warm night counts per threshold:")
threshold_results = []
for thresh in [20, 22, 24, 26]:
    yearly = count_above_threshold(nights_valid, 'night_T_min', thresh)
    if len(yearly) < 4:
        continue
    r = mk_analysis_series(
        yearly['year'].values, yearly['count'].values,
        f'Nights with T_min > {thresh} C'
    )
    print(f"\\n{r['summary']}")
    threshold_results.append(r)

print("\\nSummary table:")
print_mk_table(threshold_results)
"""
    ),
    new_md("### 14.4 Pettitt Change-Point Detection"),
    new_code(
        """\
# Pettitt test on mean T_min series
print("\\nPettitt change-point on mean T_min:")
for name, df, col in [('Sbd Nord (night)', nightly_mean, 'night_T_min'),
                       ('Sbd Centre (daily)', daily_mean, 'T_min')]:
    if col not in df.columns:
        continue
    pt = pettitt_test(df[col].values)
    cp_year = df['year'].values[int(pt['cp_index'])] if not np.isnan(pt['cp_index']) else 'N/A'
    print(f"  {name:25s}: CP at year {cp_year}, "
          f"U={pt['U_stat']:.1f}, p={pt['p_value']:.4f} "
          f"{'*' if pt['significant_0.05'] else ''}")
"""
    ),
    new_md("### 14.5 Trend in Streak Lengths"),
    new_code(
        """\
# MK on longest streak per year (T_min > threshold)
print("\\nMK on longest warm-night streaks:")
streak_results = []
for thresh in [20, 22, 24, 26]:
    streaks = compute_streaks(nights_valid, 'night_T_min', thresh)
    if len(streaks) < 4:
        continue
    r = mk_analysis_series(
        streaks['year'].values, streaks['longest_streak'].values,
        f'Longest streak > {thresh} C'
    )
    print(f"\\n{r['summary']}")
    streak_results.append(r)
print("\\nSummary table:")
print_mk_table(streak_results)
"""
    ),
]


def inject(notebook_path, cells):
    """Inject cells at the end of a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    for c in cells:
        if c.cell_type == 'code':
            src = c.source.replace('{{SCRIPTS}}', SCRIPTS)
            nb.cells.append(new_code(src))
        else:
            nb.cells.append(new_md(c.source))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"  Injected {len(cells)} cells in {os.path.basename(notebook_path)}")


if __name__ == '__main__':
    notebooks = {
        os.path.join(PROJECT, 'Seccions', 'Pluges_a_les_diferents_estacions', 'Rain_seasons.ipynb'): RAIN_CELLS,
        os.path.join(PROJECT, 'Seccions', 'Pluges_a_les_diferents_estacions', 'Hourly_rain_analysis.ipynb'): HOURLY_CELLS,
        os.path.join(PROJECT, 'Seccions', 'Pluges_a_les_diferents_estacions', 'Intensity_pluja.ipynb'): INTENSITY_CELLS,
        os.path.join(PROJECT, 'Seccions', 'Tempereatures_a_la_nit', 'Evolucio_temperatures_nocturnes.ipynb'): TEMP_CELLS,
    }

    for path, cells in notebooks.items():
        if os.path.exists(path):
            inject(path, cells)
            print(f"  Done: {os.path.basename(path)}")
        else:
            print(f"  Not found: {path}")

    print("\\nAll done! Open the notebooks to run the new statistical analysis cells.")
