import dash_ag_grid as dag
from dash import Dash, html, dcc, Input, Output, callback, Patch, State
import pandas as pd
import dash_bootstrap_components as dbc
import os
import pandas as pd
import core.misc as misc

AnalysisMode = 1
AutoCheck = False
LockPointX = 0.0
LockPointY = 0.0
currDirectory = ""
#SelectedPath = ""
selected_list = []
LogPath = "V:/LatticePhaseLock/LatticePhaseLock.txt"
phase_iteration = 0

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "30rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

columnDefs = [{"field": i} for i in ["File Name", "Date Modified", "analyzed"]]

button_group = html.Div(
    [
        dbc.RadioItems(
            id="radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Lock", "value": 1},
                {"label": "Phase X", "value": 2},
                {"label": "Phase Y", "value": 3},
            ],
            value=1,
        ),
        html.Div(id="output"),
    ],style={ 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'},
    className="radio-group",
)

phasebox_content = [html.P("X"),dbc.Input(id="input-phasex", size = "10"),
                html.P("Y"),dbc.Input(id="input-phasey", size = "10")]
sidebar = html.Div(
    [
        #html.P("Lattice Analysis", className="display-4"),
        html.Hr(),
        html.P(
            "Lattice reconstruction analysis", className="lead"
        ),
        html.Div([
            
           
#        dbc.Checklist(
 #           options=[
  #              {"label": "Auto Check", "value": 1},
   #         ],
    #        value=[],
     #       id="switches-autocheck",
      #      switch=True, style={"textAlign":"left"},
       # ), 
        
            dbc.Button("Refresh", id="button-ForceRefresh", style={"textAlign":"right"})], 
        style={'display':'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
  
        html.Div(
          [
              dbc.Input(value=currDirectory, id="input-path", placeholder="Please type path ... ", type="text",size="200"),
          ]),
        
        dag.AgGrid(
            id="table-filename",
            columnDefs=columnDefs,
            rowData=[],
            columnSize="sizeToFit",
            defaultColDef={"filter": True},
            dashGridOptions={"animateRows": False, "rowSelection":"multiple", "pagination":True, "PaginationPageSize":100}
        ),
        html.Pre(id="pre-table-filename", style={'text-wrap': 'wrap'}),
        html.Hr(),
        button_group,
        html.Div(id='div-phasebox', 
                 children = phasebox_content,
                 style={'display': 'flex', 'align-items': 'center', 'gap': '10px'}
        ),
        html.Div(id="div-empty"),
        html.Div([dbc.Button('Analysis', id='button-analysis',n_clicks=0),
                  dbc.Button('Analysis All Contents', id='button-analysis-all',n_clicks=0, color = 'secondary'),
                  dbc.Button('Auto Check', id='button-autocheck',n_clicks=0, color="warning"), 
                  dbc.Button('Stop', id='button-stop',n_clicks=0, color="danger")],
                 style={'display':'flex', 'justify-content': 'space-between', 'align-items': 'center'},),

        html.Div( [
                html.H5(dbc.Badge("Phase file name ",className="ms-1",style={"margin-light":"20px"})),
                dbc.Input(id="input-logpath", type="text", value = LogPath, style={'width':'80%'})],
                 style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),        
        html.P(id='P-phase-iteration'),
        
    ],
    style=SIDEBAR_STYLE,
)


@callback(Output("table-filename", "rowData") ,
          Input("input-path","value"), 
          Input("button-ForceRefresh", "n_clicks"),
          State("table-filename", "rowData"))
def UpdatePathTable(directory, n_clicks, data_prev) :
    global currDirectory
    try :
        if not os.path.exists(directory) :
            return data_prev    
    except:
        return data_prev
    # Get list of .txt files with modified times
    file_info = []
    with os.scandir(directory) as entries:
        file_info = [
            {"File Name": entry.name,
             "Date Modified": entry.stat().st_mtime,
             "analyzed": misc.exist_analysisfile(directory, entry.name)}
            for entry in entries if entry.is_file() and entry.name.endswith('.dat')
        ]
    print("File Processing done...")
    if(len(file_info) == 0) :
        return data_prev 
    currDirectory = directory

    # Convert to DataFrame and format modified time
    df = pd.DataFrame(file_info)
    df["Date Modified"] = pd.to_datetime(df["Date Modified"], unit='s')
    return df.to_dict("records")

@callback(Output("input-logpath","valid"),
          Input("input-logpath","value"))
def UpdateInputpath(directory) :
    try :
        if not os.path_exists(directory) :
            return False
    except :
        return False
    global LogPath
    LogPath = directory
    return True

@callback(Output('div-phasebox', 'children'), Input("radios", "value"))
def ModeSelection(value) :
    global AnalysisMode
    global LockPointX 
    global LockPointY 
    global phase_iteration
    AnalysisMode = value
    phase_iteration = 0        
    if value == 1 :
        return phasebox_content
    else :
        LockPointX, LockPointY = (0.0, 0.0)
        return []

@callback(Output('input-phasex','valid'),Input('input-phasex','value')) 
def updatephasex(valuex):
    global LockPointX
    r1 = False
    try:
        LockPointX = float(valuex)
        r1 = True
    except:
        r1 = False
    return r1

@callback(Output('input-phasey','valid'),Input('input-phasey','value'))
def updatephasey(valuey) :
    global LockPointY
    r2 = False
    try:
        LockPointY = float(valuey)
        r2 = True
    except:
        r2 = False
    return r2


@callback(
    Output("pre-table-filename", "children"),
    Input("input-path", "value"),
    Input("table-filename", "selectedRows"),
)
def output_selected_rows(directory, selected_rows):
    global selected_list
    global currDirectory
    if(type(selected_rows) == type(None)) :
        return
    if len(selected_rows) > 1 :
        selected_list = [f"{selected_rows[0]['File Name']}",f"{selected_rows[-1]['File Name']}"]
    else :
        selected_list = [f"{s['File Name']}" for s in selected_rows]
    currDirectory = directory
    return f"Selected files {'s' if len(selected_rows) > 1 else ''}:{', '.join(selected_list)}" if selected_rows else "No selections"

'''
@callback(
    Output("div-empty","children"),
    Input("switches-autocheck","value")
)
def UpdateAutoCheck(value) :
    global AutoCheck
    AutoCheck = value==1
    return []
''' 
    
    