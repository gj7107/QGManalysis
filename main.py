import dash_ag_grid as dag
from dash import Dash, html, dcc, Input, Output, callback, Patch, State, no_update
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import core.sidebar as sidebar
import core.LatticeAnalysisUnit as LatticeAnalysisUnit
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import core.misc as msic
import time
from dash.exceptions import PreventUpdate
import threading

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
unit = LatticeAnalysisUnit.LatticeAnalysisUnit()
    
Lats = []
photoncounts = []
xopts = []
phasexlog = []
phaseylog = []
timelog = []

offsetX = 0
offsetY = 0

prevfilename = ""
Latsum = 0
recent_figure_content = []
recent_phase_content = ""


FLAG_STOP = False
FLGA_REQUIRE_UPDATE = False
# the style arguments for the sidebar. We use position:fixed and a fixed width

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "32rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    'flex':'1'
}


content = html.Div(children ="No File loaded", id="page-content", style=CONTENT_STYLE)

content_top = html.Div(children = [html.Hr(),html.P(id="text-currPhase")], id = "content-top", style = {
    "margin-left":"32rem"})

app.layout = html.Div([dcc.Location(id="url"), sidebar.sidebar,content_top, content, 
                       dcc.Interval(id='interval-component',interval=500,n_intervals=0),
                       html.Div(id='task-component',children='Not triggered')])


@app.callback(
    Output("page-content","children"),
    Output("P-phase-iteration", 'children'),
    Output("text-currPhase","children"),
    Input("interval-component", "n_intervals")
  #  Input("task-component","children"),
    #State("switches-autocheck","value"),
)
def AnalysisButton(n_intervals) :
    global Lats, photoncounts, xopts, prevfilename
    global recent_figure_content, recent_phase_content
    global FLGA_REQUIRE_UPDATE

    ctx = dash.callback_context
    if not ctx.triggered : 
        trigger_id = 'No trigger yet'
    else :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    #print(trigger_id)
    
    #elif(trigger_id == 'task-component' and len(value) == 1) :
    #    Lats, photoncounts, xopts = CheckAndPhaseOutput(sidebar.currDirectory, prevfilename, sidebar.AnalysisMode)
    if(trigger_id == 'interval-component') :
        if not FLGA_REQUIRE_UPDATE :
            return no_update, no_update, no_update
    
    else : 
        print("trigger with", trigger_id)
        #return recent_figure_content, f"iteration = {sidebar.phase_iteration}", recent_phase_content
        if not FLGA_REQUIRE_UPDATE :
            return no_update, no_update, no_update
    
    px = xopts[0][0]
    py = xopts[0][1]
    print(Lats[0].shape)
    uphasex = np.unwrap(np.array(phasexlog).reshape(-1) * 2 * np.pi) / 2 / np.pi
    uphasey = np.unwrap(np.array(phaseylog).reshape(-1) * 2 * np.pi) / 2 / np.pi
    
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Heatmap(z = Lats[0]
                                    ,colorscale='viridis'),row=1,col=1)
    fig.add_trace(go.Histogram(x = Lats[0].reshape(-1),
                                nbinsx=21),row=1,col=2)
    fig.add_trace(go.Scatter(x = timelog, y = uphasex,
                                mode='lines'),row=2,col=1)
    fig.add_trace(go.Scatter(x = timelog, y = uphasey,
                                mode='lines'),row=2,col=2)
    fig.update_xaxes(title_text='Lat 0', row=1, col=1)
    fig.update_yaxes(row=1, col=1)
    fig.update_xaxes(title_text='Photon count', row=1, col=2)
    fig.update_yaxes(title_text='Count', range=[0, 1e3], row=1,col=2)
    fig.update_layout(height=800, width=1000, title_text="Analysis :" + str(prevfilename))
                        
    retval =  [html.H4("Lattice analysis"), 
            dcc.Graph(figure=fig, id='graph')]
    recent_figure_content = retval
    recent_phase_content = html.Div([html.H4("Phase X :"),
                            dbc.Badge(f"{px:.2f}", color="light", text_color="primary", className="ms-1"),
                            html.H4("Phase Y :"),
                            dbc.Badge(f"{py:.2f}", color="light", text_color="secondary", className="ms-1"),
                            ],style={'display': 'flex', 'align-items': 'center', 'gap': '10px'})
    FLGA_REQUIRE_UPDATE = False
    return recent_figure_content, f"iteration = {sidebar.phase_iteration} \n Set Offset (X,Y) = ({offsetX:.2f},{offsetY:.2f})",recent_phase_content
    
@app.callback(Output('button-analysis-all','disabled'),
              Output('button-autocheck','disabled'),
              Output('button-analysis-all','active'),
              Output('button-autocheck','active'),
              Input('button-analysis','n_clicks'),
              Input('button-analysis-all','n_clicks'),
              Input('button-autocheck','n_clicks'),
              Input('button-stop','n_clicks')
              )
def ButtonCallback(v0, v1, v2, v3) :
    global FLAG_STOP

    ctx = dash.callback_context
    if not ctx.triggered : 
        trigger_id = 'No trigger yet'
    else :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(trigger_id)
    
    if(trigger_id == 'button-analysis' ) :
        FLAG_STOP = False    
        th_anlaysis = threading.Thread(target = AnalaysisSelected, args = [])
        th_anlaysis.start()
        return no_update, no_update, no_update, no_update
    elif trigger_id == 'button-analysis-all' :
        FLAG_STOP = False    
        th_analysisall = threading.Thread(target = AnalysisAll, args = [])
        th_analysisall.start()
        return True, True, True, False
    elif trigger_id == 'button-autocheck' :
        FLAG_STOP = False
        th_autocheck = threading.Thread(target = AutoCheck, args = [])
        th_autocheck.start()
        return True, True, False, True
    elif trigger_id == 'button-stop' : 
        print("STOP!")
        FLAG_STOP = True    
        return False, False, False, False
    else :
        print(trigger_id) 
        return  False, False, False, False
    
def AnalaysisSelected():
    global Lats, photoncounts, xopts
    global FLGA_REQUIRE_UPDATE
    for selectedfile in sidebar.selected_list :
        if FLAG_STOP :
            break
        SelectedPath = os.path.join(sidebar.currDirectory, selectedfile)
        if(os.path.isfile(SelectedPath)) :
            print("Selected Path :" , SelectedPath)
            prevfilename = os.path.basename(SelectedPath)
            print("Prev file name :", prevfilename)
            #Lats, photoncounts, xopts =  unit.AnalysisOne(SelectedPath) 
            Lats, photoncounts, xopts = CheckAndPhaseOutput(sidebar.currDirectory, prevfilename, sidebar.AnalysisMode)
            FLGA_REQUIRE_UPDATE = True

def AutoCheck():
    global Lats, photoncounts, xopts, prevfilename
    global FLGA_REQUIRE_UPDATE
    while not FLAG_STOP:
        time.sleep(0.1)
        FolderPath = sidebar.currDirectory
        if(not os.path.isdir(FolderPath)) :
            continue
        
        newestfile = misc.get_newest_file(FolderPath, ".dat")
        if(newestfile == prevfilename) :
            continue
        
        prevfilename = newestfile
        print("New file found ...", newestfile)
        
        Lats, photoncounts, xopts = CheckAndPhaseOutput(FolderPath, prevfilename, sidebar.AnalysisMode)
        FLGA_REQUIRE_UPDATE = True
    return

def AnalysisAll() :
    global Lats, photoncounts, xopts, prevfilename
    global FLGA_REQUIRE_UPDATE
    files =  misc.get_all_files(sidebar.currDirectory, '.dat') 
    for file in files :
        if FLAG_STOP :
            break
        if(os.path.isfile(os.path.join(sidebar.currDirectory, file))) :
            prevfilename = os.path.basename(file)
            Lats, photoncounts, xopts = CheckAndPhaseOutput(sidebar.currDirectory,file, sidebar.AnalysisMode)
            FLGA_REQUIRE_UPDATE = True
    return



def CheckAndPhaseOutput(FolderPath, newestfile, mode):
    global phasexlog, phaseylog, timelog, offsetX, offsetY
    
    time.sleep(0.1)
    phase_iteration = sidebar.phase_iteration
    Lats, photoncounts, xopts = unit.AnalysisOne(os.path.join(FolderPath, newestfile))
    mtime = os.path.getmtime(os.path.join(FolderPath, newestfile))
    px = xopts[0][0]
    py = xopts[0][1]
    
    phasexlog.append(px)
    phaseylog.append(py)
    timelog.append(mtime)
    
    ux = np.unwrap(np.array(phasexlog) * 2 * np.pi) / 2/ np.pi
    uy = np.unwrap(np.array(phaseylog) * 2 * np.pi) / 2/ np.pi
    print(f"Mode-{mode} with iter = {phase_iteration}")
    if mode == 1:
        setphaseX = px + sidebar.LockPointX
        setphaseY = -py + sidebar.LockPointY
    elif mode == 2:
        setphaseX = np.mod(phase_iteration,20) / 10
        setphaseY = 0
        sidebar.phase_iteration += 1
    elif mode == 3:
        setphaseX = 0
        setphaseY = np.mod(phase_iteration,20) / 10
        sidebar.phase_iteration += 1
    else : 
        print("ERR")

    offsetX = 8.25 * (np.cos(unit.ang1) * setphaseX + np.sin(unit.ang1) * setphaseY)
    offsetY = 8.25 * (np.cos(unit.ang2) * setphaseX + np.sin(unit.ang2) * setphaseY)

    if(os.path.isfile(sidebar.LogPath)) :
        with open(sidebar.LogPath, 'w') as file:
            formatted_string = "OffsetX=%.2f\nOffsetY=%.2f\nPHASEX=%.2f\nPHASEY=%.2f\n"%(offsetX,offsetY,px*8.25,py*8.25)
            file.write(formatted_string)
    else :
        print(os.path.isfile(sidebar.LogPath))
        
    
        
        
    return Lats, photoncounts, xopts

'''
@app.callback(Output('task-component','children'),Input('interval-component','n_intervals'))
def AutoCheckCondition(n_intervals) :
    global prevfilename
    FolderPath = sidebar.currDirectory
    if(not os.path.isdir(FolderPath)) :
        return no_update
    
    newestfile = misc.get_newest_file(FolderPath, ".dat")
    if(newestfile == prevfilename) :
        return no_update
    prevfilename = newestfile
    print(newestfile, prevfilename)

    return "Done"
'''



if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run_server(debug=True, port=8089, host = '0.0.0.0')