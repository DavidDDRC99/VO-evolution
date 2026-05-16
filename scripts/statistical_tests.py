"""
Statistical tests for climate trend analysis.
Implements Mann-Kendall, Sen's slope, Pettitt test, and Seasonal Mann-Kendall
using scipy (no external dependencies beyond the project stack).
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings


def mann_kendall(y):
    """
    Mann-Kendall trend test using Kendall's tau.

    Parameters
    ----------
    y : array-like
        Time series data.

    Returns
    -------
    dict with keys:
        tau : Kendall's tau statistic
        p_value : two-sided p-value
        trend : 'increasing', 'decreasing', or 'no trend'
        alpha : significance level thresholds
    """
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 4:
        return {'tau': np.nan, 'p_value': np.nan, 'trend': 'insufficient data',
                'sign_0.05': False, 'sign_0.01': False, 'sign_0.001': False}

    tau, p_value = stats.kendalltau(np.arange(len(y))[mask], y[mask])

    result = {
        'tau': tau,
        'p_value': p_value,
        'trend': 'no trend',
        'sign_0.05': p_value < 0.05,
        'sign_0.01': p_value < 0.01,
        'sign_0.001': p_value < 0.001,
    }
    if tau > 0 and p_value < 0.05:
        result['trend'] = 'increasing'
    elif tau < 0 and p_value < 0.05:
        result['trend'] = 'decreasing'
    elif p_value < 0.05:
        result['trend'] = f'tau={tau:.3f} (check)'

    return result


def sens_slope(y, alpha=0.95):
    """
    Sen's slope estimator (Theil-Sen slope).

    Parameters
    ----------
    y : array-like
        Time series data.
    alpha : float
        Confidence level for the interval.

    Returns
    -------
    dict with keys:
        slope : median slope (units per time step)
        intercept : median intercept
        ci_lower : lower confidence bound
        ci_upper : upper confidence bound
    """
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    x = np.arange(len(y))[mask]
    y_clean = y[mask]
    if len(x) < 4:
        return {'slope': np.nan, 'intercept': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.theilslopes(y_clean, x, alpha=alpha)

    return {
        'slope': result.slope,
        'intercept': result.intercept,
        'ci_lower': result.low_slope,
        'ci_upper': result.high_slope,
    }


def pettitt_test(y):
    """
    Pettitt test for single change-point detection in a time series.
    Non-parametric, based on Mann-Whitney U statistic.

    Parameters
    ----------
    y : array-like
        Time series data.

    Returns
    -------
    dict with keys:
        cp_index : index of the change point (0-based)
        cp_value : approximate date/time label at change point
        U_stat : Pettitt's U statistic
        p_value : approximate two-sided p-value
        significant_0.05 : whether change point is significant at alpha=0.05
    """
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    y_clean = y[mask]
    n = len(y_clean)
    if n < 4:
        return {'cp_index': np.nan, 'cp_value': np.nan, 'U_stat': np.nan,
                'p_value': np.nan, 'significant_0.05': False}

    # Compute Pettitt's U statistic for each split point
    U_max = -np.inf
    cp = 0
    for k in range(1, n):
        # Mann-Whitney U statistic between [0:k] and [k:n]
        U = 0.0
        for i in range(k):
            for j in range(k, n):
                if y_clean[i] < y_clean[j]:
                    U += 1
                elif y_clean[i] > y_clean[j]:
                    U -= 1
        abs_U = abs(U)
        if abs_U > U_max:
            U_max = abs_U
            cp = k

    # Approximate p-value (Pettitt, 1979)
    p_value = 2.0 * np.exp(-2.0 * U_max**2 / (n**3 + n**2))
    p_value = min(1.0, p_value)

    return {
        'cp_index': cp,
        'cp_value': cp,
        'U_stat': U_max,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
    }


def seasonal_mann_kendall(df, value_col, month_col='month', year_col='year'):
    """
    Seasonal Mann-Kendall test (Hirsch et al., 1982).
    Accounts for seasonality by computing MK statistic per month
    and combining them.

    Parameters
    ----------
    df : DataFrame
        Must contain value_col, month_col, year_col.
    value_col : str
        Column name for the values to test.
    month_col : str
        Column name for month (1-12).
    year_col : str
        Column name for year.

    Returns
    -------
    dict with keys:
        S_total : combined Mann-Kendall S statistic
        var_S : variance of S
        Z : standardized test statistic
        p_value : two-sided p-value
        trend : 'increasing', 'decreasing', or 'no trend'
        monthly_results : list of per-month MK results
    """
    months = sorted(df[month_col].unique())
    S_total = 0.0
    var_S_total = 0.0
    monthly = []

    for m in months:
        sub = df[df[month_col] == m].sort_values(year_col)
        vals = sub[value_col].dropna().values
        if len(vals) < 3:
            continue
        n = len(vals)
        S = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                S += np.sign(vals[j] - vals[i])
        var_S = n * (n - 1) * (2 * n + 5) / 18.0

        # Handle ties
        ties = np.unique(vals, return_counts=True)[1]
        if len(ties) > 0:
            var_S -= np.sum(ties * (ties - 1) * (2 * ties + 5)) / 18.0

        monthly.append({'month': m, 'S': S, 'var_S': var_S, 'n': n})

        S_total += S
        var_S_total += var_S

    if var_S_total <= 0:
        return {'S_total': np.nan, 'var_S': np.nan, 'Z': np.nan,
                'p_value': np.nan, 'trend': 'insufficient data',
                'monthly_results': monthly}

    Z = (S_total - 1) / np.sqrt(var_S_total) if S_total > 0 else 0
    if S_total < 0:
        Z = (S_total + 1) / np.sqrt(var_S_total)

    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(Z)))

    trend = 'no trend'
    if Z > 0 and p_value < 0.05:
        trend = 'increasing'
    elif Z < 0 and p_value < 0.05:
        trend = 'decreasing'

    return {
        'S_total': S_total,
        'var_S': var_S_total,
        'Z': Z,
        'p_value': p_value,
        'trend': trend,
        'monthly_results': monthly,
    }


def mk_analysis_series(years, values, label='series'):
    """
    Convenience function: runs Mann-Kendall + Sen's slope on a series
    and returns a formatted summary.

    Parameters
    ----------
    years : array-like
    values : array-like
    label : str
        Description of the series.

    Returns
    -------
    dict with all results + formatted string summary
    """
    mk = mann_kendall(values)
    ss = sens_slope(values)
    pt = pettitt_test(values)

    result = {
        'label': label,
        'years': years,
        'values': values,
        # Mann-Kendall
        'mk_tau': mk['tau'],
        'mk_p_value': mk['p_value'],
        'mk_trend': mk['trend'],
        'mk_sign_0.05': mk['sign_0.05'],
        'mk_sign_0.01': mk['sign_0.01'],
        # Sen's slope
        'sens_slope': ss['slope'],
        'sens_intercept': ss['intercept'],
        'sens_ci_lower': ss['ci_lower'],
        'sens_ci_upper': ss['ci_upper'],
        # Pettitt
        'pt_cp_index': pt['cp_index'],
        'pt_cp_value': pt['cp_value'],
        'pt_U_stat': pt['U_stat'],
        'pt_p_value': pt['p_value'],
        'pt_significant': pt['significant_0.05'],
    }

    lines = [f"--- {label} ---"]
    if mk['trend'] == 'insufficient data':
        lines.append("  Insufficient data (<4 points)")
    else:
        lines.append(f"  Mann-Kendall tau = {mk['tau']:.4f}, "
                     f"p = {mk['p_value']:.4f} ({mk['trend']})")
        lines.append(f"  Sen's slope = {ss['slope']:.4f} per year "
                     f"[{ss['ci_lower']:.4f}, {ss['ci_upper']:.4f}]")
        if not np.isnan(pt['p_value']):
            cp_year = years[pt['cp_index']] if pt['cp_index'] < len(years) else 'N/A'
            lines.append(f"  Pettitt change point: index={pt['cp_index']}, "
                         f"year~{cp_year}, p={pt['p_value']:.4f} "
                         f"({'significant' if pt['significant_0.05'] else 'not significant'})")
        lines.append(f"  Significant at: alpha=0.05 {'YES' if mk['sign_0.05'] else 'no'}, "
                     f"alpha=0.01 {'YES' if mk['sign_0.01'] else 'no'}")
    result['summary'] = '\n'.join(lines)
    return result


def print_mk_table(results_list):
    """
    Print a nicely formatted table of MK results.

    Parameters
    ----------
    results_list : list of dicts from mk_analysis_series
    """
    print(f"{'Label':<45} {'tau':>8} {'MK p-value':>12} {'Slope':>10} "
          f"{'Trend':<14} {'Sig 0.05':>9} {'CP year':>8}")
    print("-" * 112)
    for r in results_list:
        label = r.get('label', '')[:44]
        tau = f"{r['mk_tau']:.4f}" if not np.isnan(r.get('mk_tau', np.nan)) else 'N/A'
        pv = f"{r['mk_p_value']:.4f}" if not np.isnan(r.get('mk_p_value', np.nan)) else 'N/A'
        slope = f"{r['sens_slope']:.4f}" if not np.isnan(r.get('sens_slope', np.nan)) else 'N/A'
        trend = r.get('mk_trend', 'N/A')[:14]
        sig = 'Y' if r.get('mk_sign_0.05', False) else 'N'
        cp = r.get('pt_cp_index', np.nan)
        cp_year = 'N/A'
        if not np.isnan(cp):
            years_arr = np.asarray(r.get('years', []))
            if int(cp) < len(years_arr):
                val = years_arr[int(cp)]
                if not np.isnan(val):
                    cp_year = str(int(val))
        print(f"{label:<45} {tau:>8} {pv:>10} {slope:>10} {trend:<14} {sig:>9} {cp_year:>8}")
