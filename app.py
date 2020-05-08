#!/usr/bin/python

import dash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_table
import dash_html_components as html
import pathlib
import re
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime


# GLOBAL VARIABLES
clear_trigger = 0


# FUNCTIONS

def get_date_columns(dataframe):
    date_regex = re.compile('\d{1,2}/\d{1,2}/\d{2,4}')
    cols = dataframe.columns
    return [c for i, c in enumerate(cols) if date_regex.match(c)]  # indexes of the date cols


def get_change_per_day(data):
    return [0] + (data[1:] - data[:-1]).tolist()


def get_log(data):
    v = np.log10(data + 1e-10)
    return v


def plot_country(country, yvals, y2vals):
    """Makes 3-panel plot from country data"""

    # Log values
    yvals_log = get_log(yvals)
    y2vals_log = get_log(y2vals)

    # Per-day change
    yvals_perday = get_change_per_day(yvals)
    y2vals_perday = get_change_per_day(y2vals)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}]
        ],
        vertical_spacing=0.03
    )

    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=yvals_perday,
            mode='lines',
            name='Daily New Confirmed Cases',
            line=dict(color='#ff112d', width=2)
        ),
        secondary_y=False,
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=y2vals_perday,
            mode='lines',
            name='Daily New Deaths (right axis)',
            line=dict(color='black', width=2)
        ),
        secondary_y=True,
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=yvals,
            mode='lines',
            name='Total Confirmed',
            line=dict(color='#0e59ef', width=2)
        ),
        secondary_y=False,
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=y2vals,
            mode='lines',
            name='Total Deaths (right axis)',
            line=dict(color='black', width=2)
        ),
        secondary_y=True,
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=yvals_log,
            mode='lines',
            name='log10(Total Confirmed)',
            line=dict(color='#0eefd9', width=2)
        ),
        secondary_y=False,
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scattergl(
            x=date_cols,
            y=y2vals_log,
            mode='lines',
            name='log10(Total Deaths) (right axis)',
            line=dict(color='black', width=2)
        ),
        secondary_y=True,
        row=3,
        col=1
    )

    fig.update_yaxes(
        range=[0, yvals_log[-1]+1],
        row=3,
        col=1
    )

    fig.update_annotations(dict(font_size=8))

    fig.update_layout(
        height=600,
        width=500,
        margin={'l':0,'b':15,'r':10},
        #height=600,
        #margin={'l':0,'b':0,'r':10},
        title=dict(
            text=f'{country}',
            y=0.9,
            x=0,
            #xanchor='left',
            #yanchor='top',
        ),

        #title_text=f'{country}',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation="h",
            #x=0.06, y=1.16)
            x=0, y=-0.2),
    )
    return fig



DTHRESH = 50 # Threshold for minimum number of cases/deaths

rootdir = pathlib.Path('.')
output_dir = rootdir / 'data'  # directory where the csv files are

csv_fpath = output_dir / 'Select_COVID_data_PEAKS.csv'

try:
    csv_fullpath = csv_fpath.resolve(strict=True)
except FileNotFoundError:
    print(f'CSV file not found: {csv_fpath}')
    raise
else:
    df = pd.read_csv(csv_fpath)

date_cols = get_date_columns(df)
last_date = date_cols[-1]
last_date_index = date_cols.index(last_date) + 1

removed_cols = ['Source', 'Last_Update_Date']

df.drop(removed_cols, axis=1, inplace=True)


deathsonly = df[df['Case_Type'] == 'Deaths']
dates = get_date_columns(df)

min_death_mask = deathsonly[dates[-1]] >= DTHRESH
keepers = deathsonly[min_death_mask]['Country_Region_Safe'].unique()
keepmask = df['Country_Region_Safe'].isin(keepers)
df_filter = df[keepmask]



# Make table for app display
entry_list = list(df_filter['Country_Region_Safe'].unique())

confirmed_mask = df_filter['Case_Type'] == 'Confirmed'
death_mask = df_filter['Case_Type'] == 'Deaths'

records = []
for i, entry in enumerate(entry_list):
        country_mask = df_filter['Country_Region_Safe'] == entry
        
        confirmed = df_filter[country_mask & confirmed_mask]
        deaths = df_filter[country_mask & death_mask]
        country_code_mask = df_filter.loc[country_mask, 'Classification_Code']
        country_smoothing_mask = df_filter.loc[country_mask, 'Smoothing']
        country_start_mask_c = df_filter.loc[country_mask, 'Start_Cases']
        country_peak_mask_c = df_filter.loc[country_mask, 'Peak_Cases']

        country_start_mask_d = df_filter.loc[country_mask, 'Start_Deaths']
        country_peak_mask_d = df_filter.loc[country_mask, 'Peak_Deaths']

        country_deaths_per_case_mask = df_filter.loc[country_mask, 'Deaths_per_Case']
        country_display = df_filter.loc[country_mask, 'Country_Region']
        datadict = {
                 'Location1': entry,
                 'Location': country_display.to_list()[-1],  
                 'Smoothing': country_smoothing_mask.to_list()[-1],
                 'Class': country_code_mask.to_list()[-1],
                 'Cases': confirmed['nCases'].item(),
                 'Start_C': country_start_mask_c.to_list()[-1],
                 'Peak_C': country_peak_mask_c.to_list()[-1],
                 'Deaths': deaths['nDeaths'].item(),
                 'Start_D': country_start_mask_d.to_list()[-1],
                 'Peak_D': country_peak_mask_d.to_list()[-1],
                 'Deaths/Cases(%)':  country_deaths_per_case_mask.to_list()[-1],           
     
        }
        records.append(datadict)

df_ratio = pd.DataFrame.from_records(
        data=records,
        columns=[
            'Location1',
            'Location',
            'Smoothing',
            'Class',
            'Cases',
            'Start_C',
            'Peak_C',
            'Deaths',
            'Start_D',
            'Peak_D',
            'Deaths/Cases(%)', 
        ]
)

#Create Dash/Flask app

app = dash.Dash(__name__)
server = app.server  #for server deployment

# app.index_string = """<!DOCTYPE html>
# <html>
#     <head>
#         <!-- Global site tag (gtag.js) - Google Analytics -->
#         <script async src="https://www.googletagmanager.com/gtag/js?id=UA-131327483-1"></script>
#         <script>
#           window.dataLayer = window.dataLayer || [];
#           function gtag(){dataLayer.push(arguments);}
#           gtag('js', new Date());

#           gtag('config', 'UA-131327483-1');
#         </script>
#         {%metas%}
#         <title>{%title%}</title>
#         {%favicon%}
#         {%css%}
#     </head>
#     <body>
#         {%app_entry%}
#         <footer>
#             {%config%}
#             {%scripts%}
#             {%renderer%}
#         </footer>
#     </body>
# </html>"""
app.scripts.config.serve_locally = False,
app.scripts.append_script({
    'external_url': 'https://www.googletagmanager.com/gtag/js?id=UA-131327483-1'
})
app.scripts.append_script({
    'external_url': 'https://cdn.jsdelivr.net/gh/lppier/lppier.github.io/gtag.js 5'
})

app.layout = html.Div(
    id="content",
    children=[

        html.Div(
            id="title",
            children=[
                html.H2(
                    'Comparative Analysis of COVID-19 by Levitt Lab ‪Stanford',
                    style={'color':  '#36393b', 
                           'font-family': 'Courier',
                           'font-weight': 'bold',
                           'font-size': '30px'
                           }
                )
            ]
        ),

        html.Button('Clear selection', id='clear-button'),

        dash_table.DataTable(

            id='datatable-interactivity-ids',

            columns=[
                  {"name": i, "id": i, "selectable": True} for i in df_ratio.columns
            ],

            data=df_ratio.to_dict('records'),

            hidden_columns=['Location1'],

            css=[{"selector": ".show-hide", "rule": "display: none"}],

            fill_width=True,

            style_header={
                         'textAlign': 'center', 
                         'font_size': '16px',
                         'backgroundColor': 'rgb(50, 50, 50)',
                         'color': 'white'
            },

            fixed_rows={ 'headers': True, 'data': 0 },

            style_cell={
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis', 
                    'maxWidth': 0,
                    'textAlign': 'center',
                    'backgroundColor': 'rgb(239,239,239)',
                    'color': 'black',
            },
            style_data={
                    #'font_family': 'cursive',
                    'font_size': '16px',
                    #'text_align': 'left'

            },

            style_table={
               'maxHeight': '400px',
               #'maxWidth': '1400px',
               'overflowY': 'scroll',
               'border': 'thin lightgrey solid'
            },

            style_cell_conditional=[

                {'if': {'column_id': 'Cases'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Deaths'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Peak_C'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Peak_D'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Start_C'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Start_D'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Class'}, 'width': '7%', 'textAlign': 'center'},
                {'if': {'column_id': 'Smoothing'}, 'width': '8%', 'textAlign': 'center'},
                {'if': {'column_id': 'Deaths/Cases(%)'}, 'width': '10%', 'textAlign': 'center'},
                {'if': {'column_id': 'Location'}, 'textAlign': 'left'},

            ],
            filter_action="native",
            sort_action="custom",
            sort_by=[],
            sort_mode="multi",
            row_selectable="multi",
            selected_rows=[],
            page_action="native",
        ),
        html.Div(
            id="text",
            children=[
                html.P('(1) Smoothing is done using the original lowess FORTRAN code (W. S. Cleveland, Bell Labs, 1985). (2) Classification Code: ‘c’ means Cases peaked, ‘C’ means at least half way down the peak, ‘d’, ‘D’ are same for Deaths.  The category order is from most complete ‘cCdD‘ to least complete ‘====‘. (3) Start Cases (Start_C) is the day the total number of cases exceed 50.  Days are counted from 22 January 2020. (4) Peak Cases (Peak_C) is day new cases peak. Within a category, data is sorted by increasing date cases peaked.',
                       style={
                           'color':  '#36393b', 
                           'font-family': 'Courier',
                           'font-size': '12px'
                       }
                )
            ]
        ),

        html.Div(
            id='plot-container',
            style={
                'width': '100%',
                'display': 'flex',
                'flex-wrap': 'wrap'
            }
        ),

    ]
)


# Callbacks

# sort
@app.callback(
    Output('datatable-interactivity-ids', "data"),

    [
        Input('datatable-interactivity-ids', "sort_by")
    ]
)
def sort_table(sort_cols):
    # sort_cols is a list of dicts
    # with colname and and order

    if not sort_cols:
        return df_ratio.to_dict('records')  # cheaper

    # Do sorting by selected columns
    sort_by_cols = []
    sort_order = []
    for col in sort_cols:
        cid = col["column_id"]
        # if cid == "Smoothing":
        #     cid = "_smoothindex"
        corder = col["direction"] == "asc"
        sort_by_cols.append(cid)
        sort_order.append(corder)

    # Sort dataframe
    df_ratio.sort_values(sort_by_cols, ascending=sort_order, inplace=True)
    return df_ratio.to_dict('records')


# clear button
@app.callback(
    Output('datatable-interactivity-ids', "selected_rows"),
    [
        Input('clear-button', 'n_clicks')
    ]
)
def clear_selection(a):
    return []

# plot
@app.callback(
    Output("plot-container", "children"),
    [
        Input('datatable-interactivity-ids', "data"),
        Input('datatable-interactivity-ids', "selected_rows"),
        Input('clear-button', 'n_clicks'),
    ],
)
def plot_country_by_smoothing(tbl_df, countries, click_clear):
    """Updates plots with options from click-row """

    fig_lst = []
    # Was the clear button clicked?
    global clear_trigger
    # is_clear = any(
    #     p['prop_id'] == 'clear-button.n_clicks'
    #     for p in dash.callback_context.triggered
    # )

    # if is_clear:
    if click_clear != clear_trigger:
        clear_trigger = click_clear
        return fig_lst

    if not countries:
        raise PreventUpdate

    for country in countries:
        #print(df_filter[country]['Country_Region_Safe'])
        countryname = tbl_df[country]['Location1']
        # print(countryname)

        country_mask = (df_filter['Country_Region_Safe'] == countryname)
        cmask = country_mask & (df_filter['Case_Type'] == 'Confirmed')
        dmask = country_mask & (df_filter['Case_Type'] == 'Deaths')

        # dates is global

        cvals = df_filter.loc[cmask][dates].values[0, :]
        dvals = df_filter.loc[dmask][dates].values[0, :]
        # print(cvals)
        # print(dvals)
        fig = plot_country(countryname, cvals, dvals)
        fig_lst.append(dcc.Graph(figure=fig))
    return fig_lst


if __name__ == '__main__':
    app.run_server(debug=False)