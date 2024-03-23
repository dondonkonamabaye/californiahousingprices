#import pygal.maps.world

import time
from tracemalloc import start

import numpy as np

from pandas import Series, DataFrame

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvas  # not needed for mpl >= 3.1

import mplleaflet
from IPython.display import IFrame

import base64
from PIL import Image
import io

import hvplot.pandas

import requests # Pour effectuer la requête
import pandas as pd # Pour manipuler les données
import datetime as dt

import param
import panel as pn

import plotly.express as px

import mysql.connector

import dash
#from sklearn.datasets import load_wine
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from alpha_vantage.timeseries import TimeSeries

from dash.dash_table import DataTable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################################################################################################################################


# train_data = pd.read_csv("housing_train_data.csv")
train_data = pd.read_csv("housing.csv")

train_data.dropna(inplace=True)

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)
train_data['median_house_value'] = np.log(train_data['median_house_value'] + 1)

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

#########################################################################################################################################################################

d_columns = [{'name': x, 'id': x}  for x in train_data.head(250).columns]
d_table = DataTable(
    columns = d_columns,
    data=train_data.head(250).to_dict('records'),
    cell_selectable=True,
    sort_action="native",
    filter_action="native",
    
    page_action="native",
    page_current=0,
    page_size=5)

#########################################################################################################################################################################


values_ocean_proximity_unique = train_data['ocean_proximity'].unique()
numeric_median_house_value = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']


def create_housing_density_heatmap(numeric_median_house_value='housing_median_age'):
    filtered_df_train_data_density_heatmap = train_data[(train_data['housing_median_age'] >= 0)]
    filtered_df_train_data_density_heatmap = filtered_df_train_data_density_heatmap.sort_values(by="housing_median_age", ascending=False).head(250)

    bar_fig = px.density_heatmap(filtered_df_train_data_density_heatmap, x="ocean_proximity", y=numeric_median_house_value , z="median_house_value", template="seaborn",
    color_continuous_scale="Viridis", title=f"Ocean Proximity vs {numeric_median_house_value}", text_auto=True)
    bar_fig.update_layout(paper_bgcolor='#e5ecf6', height=420)
    
    return bar_fig
    

# multi_select_values_ocean_proximity_density_heatmap = dcc.Dropdown(id='multi_select_values_ocean_proximity_density_heatmap', options=values_ocean_proximity_unique, value='ocean_proximity', clearable=False)
multi_select_median_house_value = dcc.Dropdown(id='multi_select_median_house_value', options=numeric_median_house_value, value='housing_median_age', clearable=False)

#########################################################################################################################################################################

values_ocean_proximity_unique = train_data['ocean_proximity'].unique()
numeric_median_house_value = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

def create_median_house_line(numeric_median_house_value='housing_median_age'):
    filtered_df_train_data_line = train_data[(train_data['housing_median_age'] >= 0)]
    filtered_df_train_data_line = filtered_df_train_data_line.sort_values(by="housing_median_age", ascending=False).head(250)

    bar_fig = px.bar(filtered_df_train_data_line, x="ocean_proximity", y=numeric_median_house_value , hover_name="ocean_proximity", template="seaborn",
    barmode="relative",hover_data="ocean_proximity", custom_data="ocean_proximity", title=f"Ocean Proximity vs {numeric_median_house_value}")
    bar_fig.update_layout(height=420)
    return bar_fig
    

multi_select_median_house_line = dcc.Dropdown(id='multi_select_median_house_line', options=numeric_median_house_value, value='housing_median_age', clearable=False)

#########################################################################################################################################################################



# print()
# print(train_data.head())
# print()


app = Dash(title="Ocean Proximity Statistics Dashboard Report")

app.layout = html.Div(
        children=[
            html.H3("Ocean Dashboard Visualization Report", style={"text-align":"center"}),
            
            html.Br(),

            dcc.Tabs
            ([
                dcc.Tab(label="TOP 250 of the Ocean for numerical values of Ocean Proximity",
                    children=
                    [
                        html.Br(),

                        html.Div( 
                            children=[
                                d_table,
                            ],
                            style={"text-align":"center ", "display": "inline-block", "width": "100%", "margin":"0 auto"} 
                        ),

                        html.Br(),

                        html.Div
                        ( 
                            children=
                            [
                                multi_select_median_house_value,
                                dcc.Graph(id='ocean_proximity_density_heatmap', figure=create_housing_density_heatmap()),
                            ],
                            style={"display": "inline-block", "width": "50%"} 
                        ),

                        html.Div
                        ( 
                            children=
                            [
                                multi_select_median_house_line,
                                dcc.Graph(id='numeric_median_house_line', figure=create_median_house_line()),
                            ],
                            style={"display": "inline-block", "width": "50%"} 
                        ),
                    ],
                )
        ]),
    ],
    style={"padding":"50px"}
)

@callback(Output('ocean_proximity_density_heatmap', "figure"), [Input('multi_select_median_house_value', "value"),], )
def update_housing_density_heatmap(numeric_median_house_value):
    return create_housing_density_heatmap(numeric_median_house_value)


@callback(Output('numeric_median_house_line', "figure"), [Input('multi_select_median_house_line', "value"), ])
def update_median_house_line(numeric_median_house_value):
    return create_median_house_line(numeric_median_house_value)

if __name__ == "__main__":
    app.run_server(debug=True)