import sys
sys.path.insert(0, r'C:\Users\David\Documents\VO project\VO-evolution\scripts')

import numpy as np
from statistical_tests import (
    mann_kendall, sens_slope, pettitt_test,
    seasonal_mann_kendall, mk_analysis_series
)
import pandas as pd


def test_mann_kendall_increasing():
    y = np.arange(20.0)
    result = mann_kendall(y)
    assert result['trend'] == 'increasing'
    assert result['sign_0.05'] == True
    assert result['tau'] > 0


def test_mann_kendall_decreasing():
    y = np.arange(20.0, 0.0, -1.0)
    result = mann_kendall(y)
    assert result['trend'] == 'decreasing'
    assert result['sign_0.05'] == True
    assert result['tau'] < 0


def test_mann_kendall_no_trend():
    rng = np.random.default_rng(42)
    y = rng.normal(10, 1, 30)
    result = mann_kendall(y)
    assert result['trend'] in ['no trend', 'insufficient data'] or not result['sign_0.05']


def test_mann_kendall_insufficient():
    y = np.array([1.0, 2.0])
    result = mann_kendall(y)
    assert result['trend'] == 'insufficient data'


def test_sens_slope_linear():
    y = np.arange(1.0, 21.0)
    result = sens_slope(y)
    assert np.isclose(result['slope'], 1.0, atol=0.1)
    assert result['ci_lower'] <= result['slope'] <= result['ci_upper']


def test_sens_slope_insufficient():
    y = np.array([1.0, 2.0])
    result = sens_slope(y)
    assert np.isnan(result['slope'])


def test_pettitt_change_point():
    y = np.concatenate([np.zeros(20), np.ones(20) * 10])
    result = pettitt_test(y)
    assert result['significant_0.05'] == True
    assert 17 <= result['cp_index'] <= 23


def test_seasonal_mann_kendall():
    np.random.seed(42)
    df = pd.DataFrame({
        'year': np.repeat(range(2000, 2020), 12),
        'month': np.tile(range(1, 13), 20),
        'value': np.random.rand(20 * 12) * 10,
    })
    result = seasonal_mann_kendall(df, 'value', 'month', 'year')
    assert 'S_total' in result
    assert 'p_value' in result
    assert 'trend' in result


def test_mk_analysis_series():
    years = np.arange(2000, 2020)
    values = np.arange(20.0)
    result = mk_analysis_series(years, values, 'test')
    assert 'summary' in result
    assert 'Increasing' in result['summary'] or 'increasing' in result['summary'].lower()
