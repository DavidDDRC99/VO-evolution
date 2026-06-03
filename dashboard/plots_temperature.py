import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils import (summer_mask, night_mask, count_above_threshold,
                   compute_streaks, expand_season, classify_night,
                   NIGHT_TYPE_COLORS, NIGHT_TYPE_ORDER,
                   MONTH_NAMES_LIST, SUMMER_CORE_DAYS, WINTER_CORE_DAYS)


def _prepare_nights(df_nord_hourly):
    df = df_nord_hourly.copy()
    df['is_summer'] = summer_mask(df['month'], df['day'])
    df['is_night'] = night_mask(df['hour'])
    df['night_date'] = df['datetime_utc'].dt.normalize()
    early = df['hour'] <= 5
    df.loc[early, 'night_date'] -= pd.Timedelta(days=1)

    df_night = df[df['is_night']].copy()
    df_night['night_month'] = df_night['night_date'].dt.month
    df_night['night_day'] = df_night['night_date'].dt.day
    df_night['is_summer_night'] = summer_mask(df_night['night_month'],
                                              df_night['night_day'])
    df_summer_night = df_night[df_night['is_summer_night']].copy()

    nights = df_summer_night.groupby('night_date').agg(
        night_T_min=('T_min', 'min'),
        night_T_max=('T_max', 'max'),
        readings_count=('T_avg', 'count'),
        year=('year', 'first'),
    ).reset_index()
    nights['is_complete'] = nights['readings_count'] >= 11
    nights = nights.dropna(subset=['night_T_min', 'night_T_max'])

    cov = nights.groupby('year').agg(
        complete=('is_complete', 'sum'),
    ).reset_index()
    cov['pct'] = cov['complete'] / SUMMER_CORE_DAYS
    valid_years = cov[cov['pct'] >= 0.70]['year'].tolist()

    nights_valid = nights[nights['is_complete'] &
                          nights['year'].isin(valid_years)].copy()
    return nights_valid, valid_years


def _prepare_daily_nord(df_nord_hourly, valid_years):
    df = df_nord_hourly.copy()
    df['is_summer'] = summer_mask(df['month'], df['day'])
    df_summer = df[df['is_summer']].copy()

    daily = df_summer.groupby(df_summer['datetime_utc'].dt.normalize()).agg(
        T_min=('T_min', 'min'),
        T_max=('T_max', 'max'),
        T_avg=('T_avg', 'mean'),
        year=('year', 'first'),
    ).reset_index()
    daily.rename(columns={'datetime_utc': 'date'}, inplace=True)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.dropna(subset=['T_min'])
    daily = daily[daily['year'].isin(valid_years)].copy()
    return daily


def _prepare_centre_summer(df_centre, threshold=0.70):
    df = df_centre.copy()
    df['is_summer'] = summer_mask(df['month'], df['day'])
    df_summer = df[df['is_summer']].copy()

    cov = df_summer.groupby('year').agg(total=('date', 'count')).reset_index()
    cov['pct'] = cov['total'] / SUMMER_CORE_DAYS
    valid_years = cov[cov['pct'] >= threshold]['year'].tolist()

    df_valid = df_summer[df_summer['year'].isin(valid_years)].copy()
    return df_valid, valid_years


def plot_warm_nights_nord(nights_valid):
    thresholds = [20, 22]
    colors = ['#e74c3c', '#e67e22']

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'Nits amb T_min > {t}°C' for t in thresholds])

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        yearly = count_above_threshold(nights_valid, 'night_T_min', thresh)
        fig.add_trace(
            go.Bar(x=yearly['year'], y=yearly['count'],
                   marker_color=color, showlegend=False,
                   hovertemplate='Any: %{x}<br>Nits: %{y}'),
            row=1, col=i + 1
        )
        if len(yearly) >= 3:
            slope, intercept, _, _, _ = stats.linregress(yearly['year'].values.astype(float),
                                                         yearly['count'].values.astype(float))
            fig.add_trace(
                go.Scatter(x=yearly['year'], y=intercept + slope * yearly['year'].astype(float),
                           mode='lines', line=dict(color='red', dash='dash'),
                           name=f'Tendència: {slope:+.2f}/any', showlegend=False),
                row=1, col=i + 1
            )

    fig.update_layout(
        title=dict(text='Evolució de Nits Càlides — Sabadell Nord', font=dict(size=16)),
        height=400, margin=dict(t=50, b=30, l=50, r=30),
    )
    fig.update_yaxes(title_text='Nombre de nits', row=1, col=1)
    fig.update_yaxes(title_text='Nombre de nits', row=1, col=2)
    fig.update_xaxes(title_text='Any', tickangle=45)
    return fig


def plot_warm_nights_centre(centre_summer):
    thresholds = [20, 22, 24, 26]
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f'Dies amb T_min > {t}°C' for t in thresholds])

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        row, col = (i // 2) + 1, (i % 2) + 1
        yearly = count_above_threshold(centre_summer, 'T_min', thresh)
        fig.add_trace(
            go.Bar(x=yearly['year'], y=yearly['count'],
                   marker_color=color, showlegend=False,
                   hovertemplate='Any: %{x}<br>Dies: %{y}'),
            row=row, col=col
        )
        if len(yearly) >= 3:
            slope, intercept, _, _, _ = stats.linregress(yearly['year'].values.astype(float),
                                                         yearly['count'].values.astype(float))
            fig.add_trace(
                go.Scatter(x=yearly['year'], y=intercept + slope * yearly['year'].astype(float),
                           mode='lines', line=dict(color='red', dash='dash'),
                           name=f'Tendència: {slope:+.2f}/any', showlegend=False),
                row=row, col=col
            )

    fig.update_layout(
        title=dict(text='Evolució de Dies Càlids — Sabadell Centre', font=dict(size=16)),
        height=600, margin=dict(t=50, b=30, l=50, r=30),
    )
    fig.update_yaxes(title_text='Nombre de dies')
    fig.update_xaxes(title_text='Any', tickangle=45)
    return fig


def plot_streaks_nord(nights_valid):
    thresholds = [20, 22]
    colors = ['#e74c3c', '#e67e22']

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'Ratxa de nits > {t}°C' for t in thresholds])

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        streaks = compute_streaks(nights_valid, 'night_T_min', thresh)
        fig.add_trace(
            go.Bar(x=streaks['year'], y=streaks['longest_streak'],
                   marker_color=color, name='Ratxa més llarga',
                   hovertemplate='Any: %{x}<br>Ratxa: %{y} dies'),
            row=1, col=i + 1
        )
        fig.add_trace(
            go.Bar(x=streaks['year'], y=streaks['n_streaks_3plus'],
                   marker_color='gray', name='Ratxes ≥ 3 dies',
                   hovertemplate='Any: %{x}<br>Ratxes ≥3: %{y}'),
            row=1, col=i + 1
        )

    fig.update_layout(
        title=dict(text='Ratxes de Nits Càlides — Sabadell Nord', font=dict(size=16)),
        height=400, margin=dict(t=50, b=30, l=50, r=30),
        barmode='group',
    )
    fig.update_yaxes(title_text='Nombre de dies')
    fig.update_xaxes(title_text='Any', tickangle=45)
    return fig


def plot_streaks_centre(centre_summer):
    thresholds = [20, 22, 24, 26]
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f'Ratxa de dies > {t}°C' for t in thresholds])

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        row, col = (i // 2) + 1, (i % 2) + 1
        streaks = compute_streaks(centre_summer, 'T_min', thresh)
        fig.add_trace(
            go.Bar(x=streaks['year'], y=streaks['longest_streak'],
                   marker_color=color, name='Ratxa més llarga',
                   hovertemplate='Any: %{x}<br>Ratxa: %{y} dies'),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(x=streaks['year'], y=streaks['n_streaks_3plus'],
                   marker_color='gray', name='Ratxes ≥ 3 dies',
                   hovertemplate='Any: %{x}<br>Ratxes ≥3: %{y}'),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text='Ratxes de Dies Càlids — Sabadell Centre', font=dict(size=16)),
        height=600, margin=dict(t=50, b=30, l=50, r=30),
        barmode='group',
    )
    fig.update_yaxes(title_text='Nombre de dies')
    fig.update_xaxes(title_text='Any', tickangle=45)
    return fig


def plot_tmin_comparison(daily_nord_valid, centre_summer):
    valid_years_nord = sorted(daily_nord_valid['year'].unique())
    valid_years_centre = sorted(centre_summer['year'].unique())
    years_common = sorted(set(valid_years_nord) & set(valid_years_centre))

    fig = go.Figure()

    for y in years_common:
        nord_data = daily_nord_valid[daily_nord_valid['year'] == y]['T_min'].dropna().values
        centre_data = centre_summer[centre_summer['year'] == y]['T_min'].dropna().values

        fig.add_trace(go.Box(
            y=nord_data, x=[y - 0.2] * len(nord_data),
            name=f'{y} Nord',
            marker_color='#3498db', boxmean=False,
            showlegend=False, width=0.3,
        ))
        fig.add_trace(go.Box(
            y=centre_data, x=[y + 0.2] * len(centre_data),
            name=f'{y} Centre',
            marker_color='#e74c3c', boxmean=False,
            showlegend=False, width=0.3,
        ))

    med_nord = [np.median(daily_nord_valid[daily_nord_valid['year'] == y]['T_min'].dropna().values)
                for y in years_common]
    med_centre = [np.median(centre_summer[centre_summer['year'] == y]['T_min'].dropna().values)
                  for y in years_common]

    slope_n, int_n, _, _, _ = stats.linregress(years_common, med_nord)
    slope_c, int_c, _, _, _ = stats.linregress(years_common, med_centre)

    fig.add_trace(go.Scatter(
        x=years_common, y=med_nord,
        mode='lines+markers', marker=dict(symbol='square', color='#2980b9', size=8),
        line=dict(color='#2980b9', width=2),
        name=f'Sbd Nord mediana ({slope_n:+.3f}°C/any)',
    ))
    fig.add_trace(go.Scatter(
        x=years_common, y=med_centre,
        mode='lines+markers', marker=dict(symbol='triangle-up', color='#c0392b', size=8),
        line=dict(color='#c0392b', width=2),
        name=f'Sbd Centre mediana ({slope_c:+.3f}°C/any)',
    ))

    fig.update_layout(
        title=dict(text="Comparació de T_min Diària per Any d'Estiu — Sbd Nord vs Sbd Centre",
                   font=dict(size=16)),
        xaxis=dict(title='Any', tickmode='array',
                   tickvals=years_common,
                   ticktext=[str(y) for y in years_common]),
        yaxis=dict(title='T_min diària (°C)'),
        height=500, margin=dict(t=50, b=30, l=50, r=30),
        hovermode='x unified',
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_summer_duration(df_daily, station_name, ref_years):
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    ref_tavg = None
    ref_data = df[(df['date'] >= f'{ref_years[0]}-01-01') &
                  (df['date'] <= f'{ref_years[-1]}-12-31')]
    for y in ref_years:
        core_start = pd.Timestamp(f'{y}-06-21')
        core_end = pd.Timestamp(f'{y}-09-21')
        core = ref_data[(ref_data['date'] >= core_start) &
                        (ref_data['date'] <= core_end)]['T_avg'].dropna()
        if len(core) >= 10:
            ref_tavg = core.mean()
            break

    if ref_tavg is None:
        return None

    results = []
    for y in range(2021, 2026):
        r = expand_season(df, y, mode='summer', target_tavg=ref_tavg)
        if r:
            results.append(r)

    if not results:
        return None

    res_df = pd.DataFrame(results)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res_df['year'], y=res_df['duration_days'],
        marker_color='#e76f51',
        text=[f'{d} dies' for d in res_df['duration_days']],
        textposition='outside',
        name='Durada total',
    ))
    fig.add_hline(y=SUMMER_CORE_DAYS, line=dict(color='gray', dash='dash'),
                  annotation_text=f'Core ({SUMMER_CORE_DAYS} dies)')

    fig.update_layout(
        title=dict(text=f'Durada Estiu — {station_name} (ref: {ref_years[0]}-{ref_years[-1]})',
                   font=dict(size=16)),
        xaxis=dict(title='Any'),
        yaxis=dict(title='Dies'),
        height=400, margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_winter_duration(df_daily, station_name, ref_years):
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    ref_tavg = None
    for y in ref_years:
        core_start = pd.Timestamp(f'{y}-12-21')
        core_end = pd.Timestamp(f'{y + 1}-03-21')
        core = df[(df['date'] >= core_start) & (df['date'] <= core_end)]['T_avg'].dropna()
        if len(core) >= 10:
            ref_tavg = core.mean()
            break

    if ref_tavg is None:
        return None

    results = []
    for y in range(2021, 2026):
        r = expand_season(df, y, mode='winter', target_tavg=ref_tavg)
        if r:
            results.append(r)

    if not results:
        return None

    res_df = pd.DataFrame(results)
    res_df['diff'] = res_df['duration_days'] - WINTER_CORE_DAYS
    colors = ['#e63946' if d < 0 else '#457b9d' for d in res_df['diff']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res_df['year'], y=res_df['duration_days'],
        marker_color=colors,
        text=[f'{d} dies' for d in res_df['duration_days']],
        textposition='outside',
        name='Durada total',
    ))
    fig.add_hline(y=WINTER_CORE_DAYS, line=dict(color='gray', dash='dash'),
                  annotation_text=f'Core ({WINTER_CORE_DAYS} dies)')

    fig.update_layout(
        title=dict(text=f'Durada Hivern — {station_name} (ref: {ref_years[0]}-{ref_years[-1]})',
                   font=dict(size=16)),
        xaxis=dict(title='Temporada'),
        yaxis=dict(title='Dies'),
        height=400, margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_night_pie(df_centre):
    df = df_centre.copy()
    df['night_type'] = df['T_min'].apply(classify_night)

    yearly = (
        df[df['night_type'].notna()]
        .groupby(['year', 'night_type'])
        .size()
        .unstack(fill_value=0)
    )
    yearly['total'] = yearly.sum(axis=1)
    valid_years = yearly[yearly['total'] >= 200].index
    yearly_filt = yearly.loc[valid_years]

    pie_years = sorted(yearly_filt.index.tolist())
    selected_years = [pie_years[0]]
    for y in [2015, 2018, 2022]:
        if y in pie_years:
            selected_years.append(y)
    if pie_years[-1] not in selected_years:
        selected_years.append(pie_years[-1])
    selected_years = sorted(set(selected_years))

    available_cats = [c for c in NIGHT_TYPE_ORDER if c in yearly_filt.columns]
    plot_colors = [NIGHT_TYPE_COLORS[c] for c in available_cats]

    fig = make_subplots(rows=1, cols=len(selected_years),
                        subplot_titles=[f'{y} (N={int(yearly_filt.loc[y, "total"])})'
                                        for y in selected_years],
                        specs=[[{'type': 'domain'}] * len(selected_years)])

    for i, year in enumerate(selected_years):
        data = yearly_filt.loc[year, available_cats].values
        fig.add_trace(
            go.Pie(labels=available_cats, values=data,
                   marker_colors=plot_colors, textinfo='percent',
                   showlegend=(i == 0),
                   legendgroup='types'),
            row=1, col=i + 1
        )

    fig.update_layout(
        title=dict(text='Distribució de Tipus de Nit (anys seleccionats)',
                   font=dict(size=16)),
        height=450, margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig
