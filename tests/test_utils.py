import sys
sys.path.insert(0, r'C:\Users\David\Documents\VO project\VO-evolution\dashboard')

import pandas as pd
import numpy as np
from utils import (
    summer_mask, winter_mask, night_mask, assign_season,
    classify_night, count_above_threshold, compute_streaks
)


def test_summer_mask():
    month = pd.Series([6, 6, 7, 9, 9, 12])
    day = pd.Series([21, 20, 15, 21, 22, 25])
    result = summer_mask(month, day)
    assert list(result) == [True, False, True, True, False, False]


def test_winter_mask():
    month = pd.Series([12, 12, 1, 3, 3, 7])
    day = pd.Series([20, 21, 15, 21, 22, 15])
    result = winter_mask(month, day)
    assert list(result) == [False, True, True, True, False, False]


def test_night_mask():
    assert night_mask(23) == True
    assert night_mask(0) == True
    assert night_mask(5) == True
    assert night_mask(6) == False
    assert night_mask(12) == False
    assert night_mask(22) == False


def test_assign_season():
    assert assign_season(12) == 'Hivern'
    assert assign_season(1) == 'Hivern'
    assert assign_season(2) == 'Hivern'
    assert assign_season(3) == 'Primavera'
    assert assign_season(6) == 'Estiu'
    assert assign_season(9) == 'Tardor'


def test_classify_night():
    assert classify_night(35) == 'Infernal (>30°C)'
    assert classify_night(28) == 'Tòrrida (>25°C)'
    assert classify_night(22) == 'Tropical (>20°C)'
    assert classify_night(15) == 'Fresca (≤20°C)'
    assert classify_night(None) is None
    assert classify_night(np.nan) is None
    assert classify_night(20) == 'Fresca (≤20°C)'
    assert classify_night(25) == 'Tropical (>20°C)'
    assert classify_night(30) == 'Tòrrida (>25°C)'


def test_count_above_threshold():
    data = pd.DataFrame({
        'year': [2020, 2020, 2020, 2021, 2021, 2021],
        'value': [10, 20, 30, 15, 25, 35],
    })
    result = count_above_threshold(data, 'value', 20)
    assert list(result['count']) == [1, 2]
    assert list(result['year']) == [2020, 2021]


def test_compute_streaks():
    data = pd.DataFrame({
        'night_date': [
            '2020-07-01', '2020-07-02', '2020-07-03',
            '2020-07-04', '2020-07-05'
        ],
        'year': [2020, 2020, 2020, 2020, 2020],
        'night_T_min': [22, 23, 18, 21, 22],
    })
    data['night_date'] = pd.to_datetime(data['night_date'])
    streaks = compute_streaks(data, 'night_T_min', 20)
    assert streaks.loc[0, 'longest_streak'] == 2
    assert streaks.loc[0, 'n_streaks_3plus'] == 0
