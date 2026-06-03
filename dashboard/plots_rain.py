import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import MONTH_NAMES, MONTH_NAMES_LIST


def plot_monthly_average(df_daily, station_name):
    monthly_totals = df_daily.groupby(['year', 'month'])['rain'].sum()
    month_rain = monthly_totals.groupby('month').mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[MONTH_NAMES[m] for m in month_rain.index],
        y=month_rain.values,
        marker_color='#2980b9',
        text=[f'{v:.1f}' for v in month_rain.values],
        textposition='outside',
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=f'Mitjana Mensual de Pluja — {station_name}', font=dict(size=16)),
        xaxis=dict(title='Mes'),
        yaxis=dict(title='Pluja (mm)'),
        height=400,
        margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_annual_evolution_grid(df_daily, station_name):
    monthly_totals = df_daily.groupby(['year', 'month'])['rain'].sum()

    fig = make_subplots(rows=4, cols=3, subplot_titles=[MONTH_NAMES[m] for m in range(1, 13)])

    for i in range(12):
        month = i + 1
        row = i // 3 + 1
        col = i % 3 + 1
        data = monthly_totals.xs(month, level='month')
        mean_val = data.mean()

        fig.add_trace(
            go.Bar(x=data.index.values, y=data.values,
                   marker_color='#2980b9', showlegend=False,
                   hovertemplate='Any: %{x}<br>Pluja: %{y:.1f} mm'),
            row=row, col=col
        )
        fig.add_hline(y=mean_val, line=dict(color='red', dash='dash'),
                      row=row, col=col)

    fig.update_layout(
        title=dict(text=f'Evolució Anual de Pluja per Mes — {station_name}', font=dict(size=16)),
        height=800,
        margin=dict(t=60, b=30, l=50, r=30),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 200])
    return fig


def plot_intensity_histogram(df_hourly, station_name):
    rain_data = df_hourly[df_hourly['rain'] > 0].copy()
    if len(rain_data) == 0:
        return None

    bins = [0, 0.2, 1, 3, 5, 10, 20, np.inf]
    labels = ['0–0.2', '0.2–1', '1–3', '3–5', '5–10', '10–20', '20+']
    rain_data['bin'] = pd.cut(rain_data['rain'], bins=bins, labels=labels, right=False)
    counts = rain_data['bin'].value_counts().reindex(labels)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts.values,
        marker_color='#3498db',
        text=[f'{int(v):,}' for v in counts.values],
        textposition='outside',
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=f'Histograma de la Intensitat de Pluja (30 min) — {station_name}',
                   font=dict(size=16)),
        xaxis=dict(title='Interval (mm)'),
        yaxis=dict(title="Nombre d'intervals"),
        height=450,
        margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_specific_rain(df_hourly, station_name):
    df_rain = df_hourly[df_hourly['rain'] > 0.2].copy()

    if 'hour_start' not in df_rain.columns:
        df_rain['hour_start'] = df_rain['datetime_utc'].dt.floor('h')

    hourly = df_rain.groupby(['year', 'hour_start'])['rain'].sum().reset_index()

    if 'Vacarisses' in station_name:
        hourly['specific_rain'] = (hourly['rain'] / 2) * 100
    else:
        hourly['specific_rain'] = (hourly['rain'] / 1) * 100

    spec_yearly = hourly.groupby('year')['specific_rain'].mean().reset_index()

    fig = go.Figure()

    colors_map = {
        'Sabadell Nord': '#3498db',
        'Sabadell Centre': '#f39c12',
        'Vacarisses': '#e74c3c',
    }
    markers_map = {
        'Sabadell Nord': 'circle',
        'Sabadell Centre': 'square',
        'Vacarisses': 'triangle-up',
    }
    c = colors_map.get(station_name, '#3498db')
    m = markers_map.get(station_name, 'circle')

    fig.add_trace(go.Scatter(
        x=spec_yearly['year'],
        y=spec_yearly['specific_rain'],
        mode='markers+lines',
        marker=dict(symbol=m, size=10, color=c),
        line=dict(dash='dash', color=c, width=1),
        name=station_name,
        hovertemplate='Any: %{x}<br>Pluja específica: %{y:.1f}',
    ))

    fig.update_layout(
        title=dict(text=f'Intensitat Mitjana de Pluja per Any — Pluja Específica ({station_name})',
                   font=dict(size=16)),
        xaxis=dict(title='Any', dtick=1),
        yaxis=dict(title='Pluja específica mitjana ((mm/h) × 100)'),
        height=450,
        margin=dict(t=50, b=30, l=50, r=30),
        hovermode='x unified',
    )
    fig.update_xaxes(tickangle=45)
    return fig
