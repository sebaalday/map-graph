#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:40:41 2019

@author: sebalday
"""
import pickle
import copy

# import importlib
# Import - GitHub
import pathlib
import os
import constants

import dash
import math
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go

import utm

#from plotly.graph_objs import *
from dash.dependencies import Input, Output
from datetime import datetime as dt
# from pyproj import Proj, transform

# App initialize - GitHub
#app = dash.Dash(__name__, 
#                meta_tags=[
#                        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
#                ]
# )
#
#server = app.server # verificar

# App initialize - Local server
app_nivel3 = dash.Dash(
        __name__
)
app_nivel3.scripts.config.serve_locally = True

app_nivel3.title = 'Pozos Pampa del Tamarugal'
app_nivel3.config["suppress_callback_exceptions"] = True


# Mapbox
mapbox_access_token = 'pk.eyJ1Ijoic2ViYXN0aWFuLWFsZGF5IiwiYSI6ImNrMzdiZ3k3bDA4aWgzY296OG5qaGducW8ifQ.hXk2h8e096wrwNWaaLrjTQ'

# Load data - Local Server
df = pd.read_csv('data/niveles_acl.csv',low_memory=False) # Base de datos niveles
coord = pd.read_csv('data/pozos_coord.csv',low_memory=False) # Ubicacion de pozos

# Load data - GitHub
#APP_PATH = str(pathlib.Path(__file__).parent.resolve())
#df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "niveles_acl.csv")))
#coord = pd.read_csv(
#    os.path.join(APP_PATH, os.path.join("data", "pozos_coord_git.csv"))
#)

# Ordenar datos. Pasar fecha en formato string a formato fecha.
df['Fecha'] = pd.to_datetime(df['Fecha'])
# Cambio coordenadas
coord['Latitude'], coord['Longitude'] = utm.to_latlon(coord['E'], coord['N'], 19, 'J') #transform(inProj, outProj, coord['E'].tolist(), coord['N'].tolist())

## Colors for legend - Local Server
#colors = [
#    "#001f3f",
#    "#0074d9",
#    "#3d9970",
#    "#111111",
#    "#01ff70",
#    "#ffdc00",
#    "#ff851B",
#    "#ff4136",
#    "#85144b",
#    "#f012be",
#    "#b10dc9",
#    "#AAAAAA",
#    "#111111",
#]

# Assign color to legend - Github    
colormap = {}
for ind, tipo in enumerate(coord["Tipo"].unique().tolist()):
    colormap[tipo] = constants.colors[ind]

# Funcion para agregar banner
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Img(src="https://www.arcadis.com/images/arcadis-logo.png?v=20191028095144"),
            html.H6("Niveles y Ubicación de Pozos"),
        ],
    )

# Titulo de grafico 
def build_graph_title(title):
    return html.P(className="graph-title", children=title)

# Helper for extracting select index from mapbox
def get_selection(data, tipo, selection_data, starting_index):
    ind = []
    current_curve = data["Tipo"].unique().tolist().index(tipo)
    for point in selection_data["points"]:
        if point["curveNumber"] - starting_index == current_curve:
            ind.append(point["pointNumber"])
    return ind


# functions
#def gen_map(map_data):
#    # groupby returns a dictionary mapping the values of the first field
#    # 'classification' onto a list of record dictionaries with that
#    # classification value.
#    return {
#        "data": [{
#                "type": "scattermapbox",
#                "lat": list(map_data['Latitude']),
#                "lon": list(map_data['Longitude']),
#                "hoverinfo": "text",
#                "hovertext": [["Name: {}".format(i)]
#                                for i in zip(map_data['Pozo'])],
#                "mode": "markers",
#                "name": list(map_data['Pozo']),
#                "marker": {
#                    "size": 6,
#                    "opacity": 0.7
#                }
#        }],
#        "layout": layout_map
#    }

# Funcion para generar ubicacion de pozos
def generate_well_map(dff, selected_data, style):
    """
    Generate well map based on selected data.
    :param dff: dataframe for generate plot.
    :param selected_data: Processed dictionary for plot generation with defined selected points.
    :param style: mapbox visual style.
    :return: Plotly figure object.
    """

    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat=-20.49, lon=-69.50),
            pitch=0,
            zoom=8,
            style=style,
        ),
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="bottom",
        ),
    )

    tipos = dff["Tipo"].unique().tolist()

    data = []

    for tipo in tipos:
        selected_index = None
        if tipo in selected_data:
            selected_index = selected_data[tipo]

        text_list = list(
            map(
                lambda item: "Well ID:" + str(int(item)),
                dff[dff["Tipo"] == tipo]["ID"],
            )
        )
        op_list = dff[dff["Tipo"] == tipo]["Operador"].tolist()
        pozo_list = dff[dff["Tipo"] == tipo]["Pozo"].tolist()

        text_list = [op_list[i] + "<br>" + pozo_list[i] + "<br>" + text_list[i] for i in range(len(text_list))]

        new_trace = go.Scattermapbox(
            lat=dff[dff["Tipo"] == tipo]["Latitude"],
            lon=dff[dff["Tipo"] == tipo]["Longitude"],
            mode="markers",
# GitHub
            marker={"color": colormap[tipo], "size": 9},
# Local Server
#            marker={"color": colormap[0], "size": 9},
            text=text_list,
            name=tipo,
            selectedpoints=selected_index,
            customdata=dff[dff["Tipo"] == tipo]["Pozo"],
        )
        data.append(new_trace)

    return {"data": data, "layout": layout}

# Layout de la app
app_nivel3.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children="Select data points from the well map to "
                                    "visualize cross-filtering to other plots. Selection could be done by "
                                    "clicking on individual data points or using the lasso tool to capture "
                                    "multiple data points or bars. With the box tool from modebar, multiple "
                                    "regions can be selected by holding the SHIFT key while clicking and "
                                    "dragging.",
                                ),
                                build_graph_title("Select Operator"),
                                dcc.Dropdown(
                                    id="operator-select",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in coord["Operador"].unique().tolist()
                                    ],
                                    multi=True,
                                    value=[
                                        coord["Operador"].unique().tolist()[0],
                                        coord["Operador"].unique().tolist()[1],
                                    ],
                                ),
                                build_graph_title("Seleccionar pozo"),
                                dcc.Dropdown(
                                        id="pozo-monitoreo-in",
                                        options=[
                                                {"label": i, "value": i}
                                                for i in df["Pozo"].unique().tolist()
                                        ],
                                        multi=True,
                                        value=[
                                                df["Pozo"].unique().tolist()[0],
                                                df["Pozo"].unique().tolist()[1],
                                        ],
                                ),
                                dcc.DatePickerRange(
                                                id='date-picker-range',
                                                min_date_allowed=dt(1900, 1, 1),
                                                max_date_allowed=dt.now(),
                                                display_format='DD/MM/YYYY',
                                                start_date=dt(1980,1, 1),
                                                end_date=dt.now()
                                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
#                        # Well map
#                        html.Div(
#                            id="well-map-container",
#                            children=[
#                                build_graph_title("Well Map"),
#                                dcc.RadioItems(
#                                    id="mapbox-view-selector",
#                                    options=[
#                                        {"label": "basic", "value": "basic"},
#                                        {"label": "satellite", "value": "satellite"},
#                                        {"label": "outdoors", "value": "outdoors"},
#                                        {
#                                            "label": "satellite-street",
#                                            "value": "mapbox://styles/mapbox/satellite-streets-v9",
#                                        },
#                                    ],
#                                    value="basic",
#                                ),
#                                dcc.Graph(
#                                    id="well-map",
#                                    figure={
#                                        "layout": {
#                                            "paper_bgcolor": "#192444",
#                                            "plot_bgcolor": "#192444",
#                                        }
#                                    },
#                                    config={"scrollZoom": True, "displayModeBar": True},
#                                ),
#                            ],
#                        ),
                        html.Div(
                                # Selected well productions
                                id="graph-container-1",
                                children=[
                                        html.Div(
                                                id="graph-header-1",
                                                children=[
                                                        build_graph_title(
                                                                "Niveles piezométricos"
                                                                ),
                                                ],
                                        ),
                                        dcc.Graph(
                                                id="graf-nivel-01-1"
                                        ),
                                ],
                        ),
                        # Ternary map
#                        html.Div(
#                            id="ternary-map-container",
#                            children=[
#                                html.Div(
#                                    id="ternary-header",
#                                    children=[
#                                        build_graph_title(
#                                            "Shale Mineralogy Composition"
#                                        ),
#                                        dcc.Checklist(
#                                            id="ternary-layer-select",
#                                            options=[
#                                                {
#                                                    "label": "Well Data",
#                                                    "value": "Well Data",
#                                                },
#                                                {
#                                                    "label": "Rock Type",
#                                                    "value": "Rock Type",
#                                                },
#                                            ],
#                                            value=["Well Data", "Rock Type"],
#                                        ),
#                                    ],
#                                ),
#                                dcc.Graph(
#                                    id="ternary-map",
#                                    figure={
#                                        "layout": {
#                                            "paper_bgcolor": "#192444",
#                                            "plot_bgcolor": "#192444",
#                                        }
#                                    },
#                                    config={
#                                        "scrollZoom": True,
#                                        "displayModeBar": False,
#                                    },
#                                ),
#                            ],
#                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # Well map
                html.Div(
                        id="well-map-container",
                        children=[
                            build_graph_title("Well Map"),
                            dcc.RadioItems(
                                id="mapbox-view-selector",
                                options=[
                                    {"label": "basic", "value": "basic"},
                                    {"label": "satellite", "value": "satellite"},
                                    {"label": "outdoors", "value": "outdoors"},
                                    {
                                        "label": "satellite-street",
                                        "value": "mapbox://styles/mapbox/satellite-streets-v9",
                                    },
                                ],
                                value="basic",
                            ),
                            dcc.Graph(
                                id="well-map",
                                figure={
                                    "layout": {
                                        "paper_bgcolor": "#192444",
                                        "plot_bgcolor": "#192444",
                                    }
                                },
                                config={"scrollZoom": True, "displayModeBar": True},
                            ),
                        ],
                    ),
                # Formation bar plots
#                html.Div(
#                    id="form-bar-container",
#                    className="six columns",
#                    children=[
#                        build_graph_title("Well count by formations"),
#                        dcc.Graph(id="form-by-bar"),
#                    ],
#                ),
                html.Div(
                    # Selected well productions
                    id="graph-container",
                    children=[
                            html.Div(
                                    id="graph-header",
                                    children=[
                                            build_graph_title(
                                                    "Niveles piezométricos"
                                                    ),
                                    ],
                            ),
                    dcc.Graph(
                            id="graf-nivel-01"
                            ),
                    ]
                ),
            ],
        ),
    ]
)
                
                # Graph
#                        html.Div(
#                            id="graph-container",
#                            children=[
#                                    html.Div(
#                                    id="graph-header",
#                                    children=[
#                                        build_graph_title(
#                                            "Niveles piezométricos"
#                                        ),
#                                    ],
#                                    ),
#                            dcc.Graph(
#                                    id="graf-nivel-01"
#                                    ),
#app.layout = html.Div(
#    children=[
#        html.Div(
#            id="top-row",
#            children=[
#                html.Div(
#                    className="row",
#                    id="top-row-header",
#                    children=[
#                        html.Div(
#                            id="header-container",
#                            children=[
#                                build_banner(),
#                                html.P(
#                                    id="instructions",
#                                    children="Visualizador de niveles Pampa del Tamarugal. Seleccione de la barra "
#                                    "el pozo que desea ver. Puede seleccionar una region con la herramienta lazo",
#                                ),
#                                build_graph_title("Selección de pozos"),
#                                dcc.Dropdown(
#                                    id="pozo-monitoreo-in",
#                                    options=[
#                                        {"label": val, "value": val}
#                                        for val in df["Pozo"].unique().tolist()
#                                    ],
#                                    multi=True,
#                                    value=[
#                                        df["Pozo"].unique().tolist()[0],
#                                        df["Pozo"].unique().tolist()[1],
#                                    ],
#                                ),
#                                dcc.DatePickerRange(
#                                    id='date-picker-range',
#                                    min_date_allowed=dt(1900, 1, 1),
#                                    max_date_allowed=dt.now(),
#                                    display_format='DD/MM/YYYY',
#                                    start_date=dt(1980,1, 1),
#                                    end_date=dt.now()
#                                ),
#                            ],
#                        ),
#                    ],
#                ),
#                html.Div(
#                    className="row",
#                    id="top-row-graphs",
#                    children=[
#                        # Well map
#                        html.Div(
#                            id="well-map-container",
#                            children=[
#                                build_graph_title("Well Map"),
#                                dcc.RadioItems(
#                                    id="mapbox-view-selector",
#                                    options=[
#                                        {"label": "basic", "value": "basic"},
#                                        {"label": "satellite", "value": "satellite"},
#                                        {"label": "outdoors", "value": "outdoors"},
#                                        {
#                                            "label": "satellite-street",
#                                            "value": "mapbox://styles/mapbox/satellite-streets-v9",
#                                        },
#                                    ],
#                                    value="basic",
#                                ),
#                                dcc.Graph(
#                                    id="well-map",
#                                    figure={
#                                        "layout": {
#                                            "paper_bgcolor": "#192444",
#                                            "plot_bgcolor": "#192444",
#                                        }
#                                    },
#                                    config={"scrollZoom": True, "displayModeBar": True},
#                                ),
#                            ],
#                        ),
#                        # Graph
#                        html.Div(
#                            id="graph-container",
#                            children=[
#                                    html.Div(
#                                    id="graph-header",
#                                    children=[
#                                        build_graph_title(
#                                            "Niveles piezométricos"
#                                        ),
#                                    ],
#                                    ),
#                            dcc.Graph(
#                                    id="graf-nivel-01"
#                                    ),
#                            ],
#                        ),
#                    ],
#                ),
#            ],
#        ),
#    ],
#)


#app.layout = html.Div(
#        
#        html.Div([
#                html.Div(
#            [
#                html.H1(children='Niveles y Ubicación de Pozos',
#                        className='nine columns'),
#                html.Img(
#                    src="https://www.arcadis.com/images/arcadis-logo.png?v=20191028095144",
#                    className='three columns',
#                    style={
#                        'height': '16%',
#                        'width': '16%',
#                        'float': 'right',
#                        'position': 'relative',
#                        'padding-top': 12,
#                        'padding-right': 0
#                    },
#                ),
#                html.Div(children='''
#                        Visualizador de los niveles de pozos Pampa del Tamarugal.
#                        ''',
#                        className='nine columns'
#                )
#            ], className="row"
#        ),
#        
#        # Selectors
#        html.Div(
#            [
#                html.Div(
#                    [
#                        html.Label('Pozo'),
#                        dcc.Dropdown(
#                                id='pozo-monitoreo-in',
#                                options=[{'label':val, 'value':val} for val in df.Pozo.unique()],
#                                value=['SNGM-PTA-0001'],
#                                multi=True
#                        ),
#                    ],
#                    className='six columns',
#                    style={'margin-top': '10'}
#                ),
#                html.Div(
#                    [
#                        dcc.DatePickerRange(
#                                id='date-picker-range',
#                                min_date_allowed=dt(1900, 1, 1),
#                                max_date_allowed=dt.now(),
#                                display_format='DD/MM/YYYY',
#                                start_date=dt(1980,1, 1),
#                                end_date=dt.now()
#                        )
#                    ],
#                    className='six columns',
#                    style={'margin-top': '10'}
#                )
#            ],
#            className='row'
#        ),
#        # Map + Graph
#        html.Div(
#            [
#                html.Div(
#                    className="row",
#                    id="top-row-graphs",
#                    children=[
#                        # Well map
#                        html.Div(
#                                id = "well-map-container",
#                                children = [
#                                        build_graph_title("Mapa de pozos"),
#                                        dcc.RadioItems(
#                                                id="mapbox-view-selector",
#                                                options=[
#                                                        {"label": "basic", "value":"basic"},
#                                                        {"label": "satellite", "value":"satellite"},
#                                                        {"label": "outdoors", "value":"outdoors"},
#                                                        {
#                                                                "label": "satellite-street",
#                                                                "value": "mapbox://styles/mapbox/satellite-streets-v9",
#                                                        },
#                                                ],
#                                                value="basic",
#                                        ),                  
#                                        dcc.Graph(
#                                                id="well-map",
#                                                figure={
#                                                     "layout": {
#                                                             "paper_bgcolor": "#192444",
#                                                             "plot_bgcolor": "#192444",
#                                                     }
#                                                },
#                                                config={"scrollZoom": True, "displayModeBar": True},
#                                        ),
#                                ],
#                                #className="six columns"
#                        ),
#                html.Div(
#                    [
#                        dcc.Graph(id='graf-nivel-01'),
#                    ],
#                    #className="six columns"
#                ),
#                html.Div(
#                    [
#                        html.P('Developed by Sebastián Alday - ', style = {'display': 'inline'}),
#                        html.A('sebastian.alday@arcadis.com', href = 'mailto:sebastian.alday@arcadis.com, sebastian.alday@ug.uchile.cl')
#                    ], className = "twelve columns",
#                       style = {'fontSize': 18, 'padding-top': 20}
#                )
#            ], #className="row"
#        )
#    ], className='ten columns offset-by-one')
#    ]))

# Funcion para crear mapa
#@app.callback(
#    Output('map-graph', 'figure'),
#    [Input('pozo-monitoreo-in', 'value')])
#def map_selection(puntos_muestreo_selected):
#    traces = pd.DataFrame()
#    aux = coord
#    for i,val_selected in enumerate(puntos_muestreo_selected):
#        df_filtered = coord[(coord['Pozo'] == val_selected)].copy()
#        traces = traces.append(df_filtered)
#    #temp_df = aux.ix[selected_row_indices, :]
#    if len(puntos_muestreo_selected) == 0:
#        return gen_map(aux)
#    return gen_map(traces)

# Update well map
@app_nivel3.callback(
    Output("well-map", "figure"),
    [
#        Input("ternary-map", "selectedData"),
#        Input("form-by-bar", "selectedData"),
#        Input("form-by-bar", "clickData"),
        Input("pozo-monitoreo-in", "value"),
        Input("mapbox-view-selector", "value"),
    ],
)
def update_well_map(
    pozo_select, mapbox_view
):
    dff = coord[coord["Pozo"].isin(pozo_select)]
    tipos = coord["Tipo"].unique().tolist()

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    processed_data = {"well_id": [], "tipo":[]}

#    if prop_id == "ternary-map":
#        for tipo in tipos:
#            if tern_selected_data is None:
#                processed_data[tipo] = None
#            else:
    for tipo in tipos:
        processed_data["well_id"] = pozo_select
        processed_data["tipo"] = tipo

#    elif prop_id == "form-by-bar":
#
#        bar_data = ""
#        if prop_type == "selectedData":
#            bar_data = bar_selected_data
#        elif prop_type == "clickData":
#            bar_data = bar_click_data
#
#        processed_data = get_selection_by_bar(bar_data)
#
#        for formation in formations:
#            if bar_data is None:
#                processed_data[formation] = None
#            elif formation not in processed_data:
#                processed_data[formation] = []

#    else:
#        for tipo in tipos:
#            processed_data[tipo] = None

    return generate_well_map(dff, processed_data, mapbox_view)


# Funcion para crear grafico
@app_nivel3.callback(
     Output('graf-nivel-01', 'figure'),
     [Input('pozo-monitoreo-in', 'value'),
      Input('date-picker-range','start_date'),
      Input('date-picker-range','end_date')])
def update_figure(puntos_muestreo_selected,s_date,e_date):
    traces = []
    #Continuar desde aca
    for i,val_selected in enumerate(puntos_muestreo_selected):
        df_filtered = df[(df['Pozo'] == val_selected)].copy()
        df_filtered=df_filtered.sort_values(by='Fecha').reset_index(drop=True)
        x1=df_filtered['Fecha']
        y1=df_filtered['Nivel']
#        pos_nan=np.where(np.isnan(y1.values))          
#        x2=x1[pos_nan].copy()
#        y2=np.zeros(x2.shape)            
        traces.append(go.Scatter(
            x=x1,
            y=y1,
            mode='markers',
            opacity=0.7,
            marker={
                'size': 8,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=val_selected
        ))
    if len(traces)==0:
        traces.append(go.Scatter(
            x=np.array([s_date,e_date]),
            y=np.nan*np.array([0,0]),
            mode='markers',
            opacity=0.7,
            marker={
                'size': 8,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Sin pozo'
        ))
        
    return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'range': [s_date,e_date]},
                yaxis={'title': 'Nivel (msnm)'},
#                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }

# Funcion para crear grafico
@app_nivel3.callback(
     Output('graf-nivel-01-1', 'figure'),
     [Input('pozo-monitoreo-in', 'value'),
      Input('date-picker-range','start_date'),
      Input('date-picker-range','end_date')])
def update_figure_1(puntos_muestreo_selected,s_date,e_date):
    traces = []
    #Continuar desde aca
    for i,val_selected in enumerate(puntos_muestreo_selected):
        df_filtered = df[(df['Pozo'] == val_selected)].copy()
        df_filtered=df_filtered.sort_values(by='Fecha').reset_index(drop=True)
        x1=df_filtered['Fecha']
        y1=df_filtered['Nivel']
#        pos_nan=np.where(np.isnan(y1.values))          
#        x2=x1[pos_nan].copy()
#        y2=np.zeros(x2.shape)            
        traces.append(go.Scatter(
            x=x1,
            y=y1,
            mode='markers',
            opacity=0.7,
            marker={
                'size': 8,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=val_selected
        ))
    if len(traces)==0:
        traces.append(go.Scatter(
            x=np.array([s_date,e_date]),
            y=np.nan*np.array([0,0]),
            mode='markers',
            opacity=0.7,
            marker={
                'size': 8,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Sin pozo'
        ))
        
    return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'range': [s_date,e_date]},
                yaxis={'title': 'Nivel (msnm)'},
#                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
 

if __name__ == '__main__':
    app_nivel3.run_server(debug=True)
