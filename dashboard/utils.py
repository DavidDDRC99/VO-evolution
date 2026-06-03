import pandas as pd
import numpy as np

MONTH_NAMES = {
    1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Oct', 11: 'Nov', 12: 'Des'
}
MONTH_NAMES_LIST = ['Gen', 'Feb', 'Mar', 'Abr', 'Mai', 'Jun',
                    'Jul', 'Ago', 'Set', 'Oct', 'Nov', 'Des']

SUMMER_CORE_DAYS = 93
WINTER_CORE_DAYS = 91


def summer_mask(month, day):
    return (
        ((month == 6) & (day >= 21)) |
        (month.isin([7, 8])) |
        ((month == 9) & (day <= 21))
    )


def winter_mask(month, day):
    return (
        ((month == 12) & (day >= 21)) |
        (month.isin([1, 2])) |
        ((month == 3) & (day <= 21))
    )


def night_mask(hour):
    return (hour >= 23) | (hour <= 5)


def assign_season(month):
    if month in [12, 1, 2]:
        return 'Hivern'
    elif month in [3, 4, 5]:
        return 'Primavera'
    elif month in [6, 7, 8]:
        return 'Estiu'
    else:
        return 'Tardor'


def classify_night(t_min):
    if pd.isna(t_min):
        return None
    if t_min > 30:
        return 'Infernal (>30°C)'
    elif t_min > 25:
        return 'Tòrrida (>25°C)'
    elif t_min > 20:
        return 'Tropical (>20°C)'
    else:
        return 'Fresca (≤20°C)'


NIGHT_TYPE_COLORS = {
    'Fresca (≤20°C)': '#3a86ff',
    'Tropical (>20°C)': '#ff9e00',
    'Tòrrida (>25°C)': '#e63946',
    'Infernal (>30°C)': '#6a040f',
}
NIGHT_TYPE_ORDER = ['Fresca (≤20°C)', 'Tropical (>20°C)',
                    'Tòrrida (>25°C)', 'Infernal (>30°C)']


def count_above_threshold(data, col, threshold):
    yearly = data.groupby('year').apply(
        lambda g: (g[col] > threshold).sum()
    ).reset_index()
    yearly.columns = ['year', 'count']
    return yearly


def compute_streaks(data, col, threshold):
    results = []
    for year, group in data.groupby('year'):
        group = group.sort_values('night_date' if 'night_date' in group.columns else 'date')
        above = (group[col] > threshold).astype(int).values

        if len(above) == 0:
            results.append({'year': year, 'longest_streak': 0, 'n_streaks_3plus': 0})
            continue

        longest = 0
        current = 0
        streaks_3plus = 0

        for val in above:
            if val == 1:
                current += 1
            else:
                if current >= 3:
                    streaks_3plus += 1
                longest = max(longest, current)
                current = 0
        if current >= 3:
            streaks_3plus += 1
        longest = max(longest, current)

        results.append({'year': year, 'longest_streak': longest,
                        'n_streaks_3plus': streaks_3plus})
    return pd.DataFrame(results)


def expand_season(df, year, mode='summer', target_tavg=None, min_core_days=10):
    if mode == 'summer':
        core_start = pd.Timestamp(f'{year}-06-21')
        core_end = pd.Timestamp(f'{year}-09-21')
        data_start = pd.Timestamp(f'{year}-05-01')
        data_end = pd.Timestamp(f'{year}-10-31')
    else:
        core_start = pd.Timestamp(f'{year}-12-21')
        core_end = pd.Timestamp(f'{year + 1}-03-21')
        data_start = pd.Timestamp(f'{year}-11-01')
        data_end = pd.Timestamp(f'{year + 1}-04-30')

    season_data = df[(df['date'] >= data_start) & (df['date'] <= data_end)] \
        .sort_values('date').reset_index(drop=True).copy()

    if len(season_data) < 30:
        return None

    core_mask = (season_data['date'] >= core_start) & (season_data['date'] <= core_end)
    if core_mask.sum() < min_core_days:
        return None

    core_idx = np.where(core_mask)[0]
    left_pos, right_pos = int(core_idx[0]), int(core_idx[-1])

    core_tavg = season_data.loc[core_idx, 'T_avg'].dropna()
    if len(core_tavg) < min_core_days:
        return None

    if target_tavg is None:
        target_tavg = core_tavg.mean()

    while left_pos > 0 and right_pos < len(season_data) - 1:
        window = season_data.iloc[left_pos - 1:right_pos + 2]['T_avg'].dropna()
        tavg = window.mean()
        if mode == 'summer' and tavg < target_tavg:
            break
        if mode == 'winter' and tavg > target_tavg:
            break
        left_pos -= 1
        right_pos += 1

    window_final = season_data.iloc[left_pos:right_pos + 1]['T_avg'].dropna()

    return {
        'year': year,
        'duration_days': len(window_final),
        'core_days': len(core_tavg),
        'window_tavg': window_final.mean(),
        'target_tavg': target_tavg,
    }
