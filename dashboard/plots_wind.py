import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import MONTH_NAMES


def compute_daily_max_wind(df_nord_hourly):
    df = df_nord_hourly.dropna(subset=['wind_avg']).copy()
    idx = df.groupby('date')['wind_avg'].idxmax()
    daily = df.loc[idx, ['date', 'year', 'month', 'hour', 'minute',
                          'wind_avg', 'wind_max', 'wind_dir']].copy()
    daily.rename(columns={'wind_avg': 'daily_max_wind'}, inplace=True)
    daily = daily.reset_index(drop=True)
    return daily


def compute_top20(daily_max):
    parts = []
    for year, group in daily_max.groupby('year'):
        parts.append(group.nlargest(20, 'daily_max_wind'))
    top20 = pd.concat(parts).reset_index(drop=True)
    return top20


def plot_wind_boxplot(top20):
    years_sorted = sorted(top20['year'].unique())
    box_data = [top20[top20['year'] == y]['daily_max_wind'].values
                for y in years_sorted]
    medians = [np.median(d) for d in box_data]

    fig = go.Figure()

    for y, d in zip(years_sorted, box_data):
        fig.add_trace(go.Box(
            y=d, name=str(y),
            boxmean=False,
            marker_color='#3498db',
            line=dict(color='#2980b9'),
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[str(y) for y in years_sorted],
        y=medians,
        mode='lines+markers',
        marker=dict(color='red', size=8, symbol='circle'),
        line=dict(color='red', width=2),
        name='Mediana',
    ))

    fig.update_layout(
        title=dict(text='Top 20 Dies Més Ventats per Any — Sabadell Nord',
                   font=dict(size=16)),
        xaxis=dict(title='Any'),
        yaxis=dict(title='Velocitat del vent mitjà (km/h) — pic diari'),
        height=500,
        margin=dict(t=50, b=30, l=50, r=30),
        hovermode='x unified',
        boxgap=0.3,
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_wind_hour_histogram(top20):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=top20['hour'],
        xbins=dict(start=-0.5, end=23.5, size=1),
        marker_color='steelblue',
        hovertemplate='Hora: %{x}<br>Pics: %{y}',
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text='Distribució Horària dels Pics de Vent — Sabadell Nord',
                   font=dict(size=16)),
        xaxis=dict(title='Hora del dia (UTC)', tickvals=list(range(24)),
                   ticktext=[f'{h}:00' for h in range(24)]),
        yaxis=dict(title="Nombre de pics de vent (top 20)"),
        height=400,
        margin=dict(t=50, b=30, l=50, r=30),
        bargap=0.05,
    )
    return fig


def plot_wind_hour_heatmap(top20):
    pivot = top20.pivot_table(index='year', columns='hour',
                              aggfunc='size', fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'{h}:00' for h in pivot.columns],
        y=[str(int(y)) for y in pivot.index],
        colorscale='YlOrRd',
        hovertemplate='Any: %{y}<br>Hora: %{x}<br>Dies: %{z}',
    ))

    fig.update_layout(
        title=dict(text='Hora del Pic de Vent per Any — Sabadell Nord',
                   font=dict(size=16)),
        xaxis=dict(title='Hora del dia (UTC)'),
        yaxis=dict(title='Any'),
        height=450,
        margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_wind_month_heatmap(top20):
    pivot = top20.pivot_table(index='year', columns='month',
                              aggfunc='size', fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[MONTH_NAMES[m] for m in pivot.columns],
        y=[str(int(y)) for y in pivot.index],
        colorscale='YlOrRd',
        hovertemplate='Any: %{y}<br>Mes: %{x}<br>Dies: %{z}',
    ))

    fig.update_layout(
        title=dict(text='Distribució Mensual dels Top 20 Dies Ventats — Sabadell Nord',
                   font=dict(size=16)),
        xaxis=dict(title='Mes'),
        yaxis=dict(title='Any'),
        height=450,
        margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig


def plot_wind_monthly_bars(top20):
    month_counts = top20.groupby('month').size()
    max_count = month_counts.max()
    colors = [f'rgba(255, {int(200*(1-v/max_count))}, 0, 0.8)'
              for v in month_counts.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[MONTH_NAMES[m] for m in month_counts.index],
        y=month_counts.values,
        marker_color=colors,
        text=month_counts.values,
        textposition='outside',
        hovertemplate='Mes: %{x}<br>Dies: %{y}',
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text='Distribució Mensual dels Dies Ventats — Sabadell Nord',
                   font=dict(size=16)),
        xaxis=dict(title='Mes'),
        yaxis=dict(title="Nombre de dies (top 20 de tots els anys)"),
        height=400,
        margin=dict(t=50, b=30, l=50, r=30),
    )
    return fig
