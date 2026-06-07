import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from data_loader import load_all_data
from plots_rain import (plot_monthly_average, plot_annual_evolution_grid,
                        plot_intensity_histogram, plot_specific_rain)
from plots_temperature import (_prepare_nights, _prepare_daily_nord,
                               _prepare_centre_summer,
                               plot_warm_nights_nord, plot_warm_nights_centre,
                               plot_streaks_nord, plot_streaks_centre,
                               plot_tmin_comparison,
                               plot_summer_duration, plot_winter_duration,
                               plot_night_pie)
from plots_wind import (compute_daily_max_wind, compute_top20,
                        plot_wind_boxplot, plot_wind_hour_histogram,
                        plot_wind_hour_heatmap, plot_wind_month_heatmap,
                        plot_wind_monthly_bars)

print("Carregant dades...")
data = load_all_data()

print("Preprocessant temperatura...")
nights_valid, valid_years_nord = _prepare_nights(data['nord_hourly'])
daily_nord_valid = _prepare_daily_nord(data['nord_hourly'], valid_years_nord)
centre_summer, valid_years_centre = _prepare_centre_summer(data['centre_daily'])

print("Preprocessant vent...")
daily_max_wind = compute_daily_max_wind(data['nord_hourly'])
top20 = compute_top20(daily_max_wind)

print("Dades carregades correctament.")

app = dash.Dash(__name__)
app.title = "VO-evolution Dashboard"

RAIN_STATIONS = ['Sabadell Centre', 'Sabadell Nord', 'Vacarisses']
TEMP_STATIONS = ['Sabadell Nord', 'Sabadell Centre']

app.layout = html.Div([
    html.H1("VO-evolution Dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50',
                   'fontSize': 32, 'marginBottom': 20}),

    dcc.Tabs(id='tabs', value='tab-rain', children=[

        dcc.Tab(label='🌧 Pluja', value='tab-rain', children=[
            html.Div([
                html.Div([
                    html.Label('Selecciona estació:',
                               style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='rain-station',
                        options=[{'label': s, 'value': s} for s in RAIN_STATIONS],
                        value='Sabadell Nord',
                        clearable=False,
                        style={'width': 300},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'margin': '15px 0', 'padding': '0 10px'}),
                html.Div(id='rain-graphs'),
            ]),
        ]),

        dcc.Tab(label='🌡 Temperatures', value='tab-temp', children=[
            html.Div([
                html.Div([
                    html.Label('Selecciona estació:',
                               style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='temp-station',
                        options=[{'label': s, 'value': s} for s in TEMP_STATIONS],
                        value='Sabadell Nord',
                        clearable=False,
                        style={'width': 300},
                    ),
                    html.Div([
                        dcc.Checklist(
                            id='show-pie',
                            options=[{'label': ' Mostrar pie chart de tipus de nit',
                                      'value': 'show'}],
                            value=[],
                            style={'marginLeft': 30},
                        ),
                    ]),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'margin': '15px 0', 'padding': '0 10px'}),
                html.Div(id='temp-graphs'),
            ]),
        ]),

        dcc.Tab(label='💨 Vent', value='tab-wind', children=[
            html.Div(id='wind-graphs'),
        ]),

    ]),
])


@app.callback(
    Output('rain-graphs', 'children'),
    Input('rain-station', 'value'),
)
def update_rain(station):
    if station == 'Sabadell Centre':
        df_daily = data['centre_daily']
        df_hourly = None
    elif station == 'Sabadell Nord':
        df_daily = data['nord_daily']
        df_hourly = data['nord_hourly']
    else:
        df_daily = data['vac_daily']
        df_hourly = data['vac_hourly']

    graphs = []

    fig1 = plot_monthly_average(df_daily, station)
    graphs.append(html.Div(dcc.Graph(figure=fig1), style={'margin': '10px 0'}))

    fig2 = plot_annual_evolution_grid(df_daily, station)
    graphs.append(html.Div(dcc.Graph(figure=fig2), style={'margin': '10px 0',
                                                           'overflowX': 'auto'}))

    if df_hourly is not None:
        fig3 = plot_intensity_histogram(df_hourly, station)
        if fig3 is not None:
            graphs.append(html.Div(dcc.Graph(figure=fig3),
                                   style={'margin': '10px 0'}))

        fig4 = plot_specific_rain(df_hourly, station)
        graphs.append(html.Div(dcc.Graph(figure=fig4), style={'margin': '10px 0'}))

    return graphs


@app.callback(
    Output('temp-graphs', 'children'),
    Input('temp-station', 'value'),
    Input('show-pie', 'value'),
)
def update_temp(station, show_pie):
    graphs = []

    if station == 'Sabadell Nord':
        fig_warm = plot_warm_nights_nord(nights_valid)
        fig_streaks = plot_streaks_nord(nights_valid)
    else:
        fig_warm = plot_warm_nights_centre(centre_summer)
        fig_streaks = plot_streaks_centre(centre_summer)

    graphs.append(html.Div(dcc.Graph(figure=fig_warm), style={'margin': '10px 0'}))
    graphs.append(html.Div(dcc.Graph(figure=fig_streaks), style={'margin': '10px 0'}))

    fig_comp = plot_tmin_comparison(daily_nord_valid, centre_summer)
    graphs.append(html.Div(dcc.Graph(figure=fig_comp), style={'margin': '10px 0'}))

    if station == 'Sabadell Nord':
        ref_years = [2009, 2010, 2011, 2012, 2013]
        fig_su = plot_summer_duration(data['nord_daily'], station, ref_years)
        fig_wi = plot_winter_duration(data['nord_daily'], station, ref_years)
    else:
        ref_years_centre = [2009, 2010, 2011, 2012, 2013]
        fig_su = plot_summer_duration(data['centre_daily'], station, ref_years_centre)
        fig_wi = plot_winter_duration(data['centre_daily'], station, ref_years_centre)

    if fig_su is not None:
        graphs.append(html.Div(dcc.Graph(figure=fig_su), style={'margin': '10px 0'}))
    if fig_wi is not None:
        graphs.append(html.Div(dcc.Graph(figure=fig_wi), style={'margin': '10px 0'}))

    if station == 'Sabadell Centre' and 'show' in show_pie:
        fig_pie = plot_night_pie(data['centre_daily'])
        graphs.append(html.Div(dcc.Graph(figure=fig_pie), style={'margin': '10px 0'}))

    return graphs


@app.callback(
    Output('wind-graphs', 'children'),
    Input('tabs', 'value'),
)
def update_wind(tab):
    if tab != 'tab-wind':
        return dash.no_update

    graphs = []

    fig1 = plot_wind_boxplot(top20)
    graphs.append(html.Div(dcc.Graph(figure=fig1), style={'margin': '10px 0'}))

    fig2 = plot_wind_hour_histogram(top20)
    graphs.append(html.Div(dcc.Graph(figure=fig2), style={'margin': '10px 0'}))

    fig3 = plot_wind_hour_heatmap(top20)
    graphs.append(html.Div(dcc.Graph(figure=fig3), style={'margin': '10px 0'}))

    fig4 = plot_wind_month_heatmap(top20)
    graphs.append(html.Div(dcc.Graph(figure=fig4), style={'margin': '10px 0'}))

    fig5 = plot_wind_monthly_bars(top20)
    graphs.append(html.Div(dcc.Graph(figure=fig5), style={'margin': '10px 0'}))

    return graphs


if __name__ == '__main__':
    app.run(debug=True)
