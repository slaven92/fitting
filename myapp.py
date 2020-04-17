import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import fittingScript as fs

from numpy.random import normal
import numpy as np
import os
from convertDataToJson import Convert
import tkinter
from tkinter.filedialog import askopenfilename
import json


def lorentzian( x, bg, x0, area, gamma):
    return  bg + area * gamma/(2*np.pi) / ( (gamma/2)**2 + ( x - x0 )**2)

def two_lorentzians(x, bg, x0, area, gamma, x02, area2, gamma2):
    return lorentzian(x, bg, x0, area, gamma) + lorentzian(x,0, x02, area2, gamma2)


root = tkinter.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
file_location = askopenfilename(master = root) # show an "Open" dialog box and return the path to the selected file

dirname = os.path.dirname(file_location)
filename = os.path.splitext(os.path.basename(file_location))[0]
json_file_location = filename + '.json'
json_file_location = os.path.join(dirname, json_file_location)
json_avr_file_location = filename + '_avr.json'
json_avr_file_location = os.path.join(dirname, json_avr_file_location)

if not os.path.exists(json_file_location) or not os.path.exists(json_avr_file_location):
    Convert(file_location).dump()

fit = fs.TwoPeaksFit(file_location)

# x = np.linspace(0, 100, 1000)
# y = two_lorentzians(x, 10, 30, 10, 10, 70, 5, 10)

peak_data = {
    'center':50,
    'bg':10,
    'peak_height':1,
    'width':20,
    'is_saved':False,
    'center2':50,
    'peak_height2':1,
    'width2':20,
    'is_saved2':False,
    'is_second' : False,
}
external_stylesheets = ['https://codepen.io/slaven92/pen/zYvGzyx.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.Div('click center max', id='mode'),

    dcc.Store(id='mem', data = peak_data), 
    
    dcc.Graph(id='mygraph', figure={
                                'data':[{
                                    'mode' : 'lines', #lines+markers, markers
                                    'name' : 'orig',
                                    },
                                    {
                                    'mode' : 'lines', #lines+markers, markers
                                    'name' : 'fit',
                                    },
                                ],
                            }
    ),

    dcc.Input(
        id='threshold',
        placeholder='Enter threshold',
        type='number',
        value='70'
    ),

    html.Button('Fit averaged data', id='fit_button'),# do fit
    
    dcc.Loading(
        id="loading-2",
        children=[html.Div(id='averaged_results', className='container')],
        type="circle",
    )

])


@app.callback(
    [Output('mem', 'data'),
    Output('mode', 'children')],
    [Input('mygraph', 'clickData')],
    [State('mem', 'data')]
)
def click_data(clickData, mem):
    text = 'click center1'
    if clickData:
        if not mem['is_second']:
            if not mem['is_saved']:     
                mem['center'] = clickData['points'][0]['x']
                mem['peak_height'] = clickData['points'][0]['y']
                mem['is_saved'] = True
                text = 'click at FWHM1'
            else:
                freq = mem['center']
                fwhm = 2 * max(abs(clickData['points'][0]['x'] - freq), 1)
                area = (mem['peak_height'] - clickData['points'][0]['y'])*np.pi*fwhm
                bg = mem['peak_height'] - 2*area/np.pi/fwhm
                mem['bg'] = bg
                mem['peak_height'] = area
                mem['width'] = fwhm
                mem['is_saved'] = False
                mem['is_second'] = True
                text = 'click center2'
        else:
            if not mem['is_saved2']:     
                mem['center2'] = clickData['points'][0]['x']
                mem['peak_height2'] = clickData['points'][0]['y']
                mem['is_saved2'] = True
                text = 'click at FWHM2'
            else:
                freq = mem['center2']
                fwhm = 2 * max(abs(clickData['points'][0]['x'] - freq), 1)
                area = (mem['peak_height2'] - clickData['points'][0]['y'])*np.pi*fwhm
                # bg = mem['peak_height2'] - 2*area/np.pi/fwhm
                # mem['bg2'] = bg
                mem['peak_height2'] = area
                mem['width2'] = fwhm
                mem['is_saved2'] = False
                mem['is_second'] = False
                text = 'click center1'
    # print(mem)
    return mem, text

@app.callback(
    Output('mygraph', 'figure'),
    [Input('mygraph', 'hoverData')],
    [State('mem', 'data'),
    State('mygraph', 'figure')]
)
def hover_data(hoverData, mem, figure):
    ctx = dash.callback_context
    num_of_elements = 450
    total = len(fit.averagedData[0]['x'])
    n = total//num_of_elements
    n_first = 0 if fit.averagedData[0]['t']>fit.averagedData[-1]['t'] else -1

    if not ctx.triggered:
        figure['data'][0]['x'] = fit.averagedData[n_first]['x'][0::n]
        figure['data'][0]['y'] = fit.averagedData[n_first]['y'][0::n]
        figure['data'][1]['x'] = fit.averagedData[n_first]['x'][0::n]
        figure['data'][1]['y'] = fit.averagedData[n_first]['y'][0::n]
        return figure

    if hoverData:
        x = figure['data'][0]['x']
        if not mem['is_second']:
            if mem['is_saved']:
                freq = mem['center']
                fwhm = 2 * max(abs(hoverData['points'][0]['x'] - freq), 1)
                area = (mem['peak_height'] - hoverData['points'][0]['y'])*np.pi*fwhm
                bg = mem['peak_height'] - 2*area/np.pi/fwhm
                
                y = lorentzian(np.array(x), bg, freq, area, fwhm)
                
                figure['data'][1]['y'] = y
        else:
            if mem['is_saved2']:
                freq = mem['center2']
                fwhm = 2 * max(abs(hoverData['points'][0]['x'] - freq), 1)
                area = (mem['peak_height2'] - hoverData['points'][0]['y'])*np.pi*fwhm
                # bg = mem['peak_height2'] - 2*area/np.pi/fwhm
                
                y = two_lorentzians(np.array(x), mem['bg'],
                        mem['center'], mem['peak_height'], mem['width'],
                        freq, area, fwhm)
                
                figure['data'][1]['y'] = y
    
    return figure


@app.callback(
    Output('averaged_results', 'children'),
    [Input('fit_button', 'n_clicks')],
    [State('threshold', 'value'),
    State('mem', 'data')]
)
def make_Fit(click, threshold, mem):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""


    initParamDict = {
        'bg' : mem['bg']*1e14,
        'center': mem['center'],
        'sigma': mem['width'],
        'amplitude': mem['peak_height']*1e14,
        'center2': mem['center2'],
        'sigma2': mem['width2'],
        'amplitude2': mem['peak_height2']*1e14,
        'split': mem['center2']-mem['center'],
    }
    # print(initParamDict)
    fit.setInitParam(initDict=initParamDict, reverse=None, threshold=int(threshold), plotData = False)
    class_name = 'one-eight column'
    ff = fit.init_params_per_temp
    num_columns = 8
    graphs = []
    inner = []
    num_of_elements = 250
    total = len(fit.averagedData[0]['x'])
    n = total//num_of_elements
    if click:
        for i, data in enumerate(fit.averagedData):

            if not i%num_columns:
                graphs.append(html.Div(inner.copy(), className='row'))
                inner=[]
            inner.append(
                    dcc.Graph(
                        figure={
                        'data':[{
                            'x' : data['x'][0::n],
                            'y' : data['y'][0::n],
                            'mode' : 'lines', #lines+markers, markers
                            'name' : 'orig',
                        },
                        {
                            'x' : data['x'][0::n],
                            'y' : two_lorentzians(np.array(data['x'][0::n]), ff[i]['bg'], ff[i]['center'],ff[i]['amplitude']*np.pi*ff[i]['sigma']/2,ff[i]['sigma'],ff[i]['center2'],ff[i]['amplitude2']*np.pi*ff[i]['sigma']/2,ff[i]['sigma2']),
                            'mode' : 'lines', #lines+markers, markers
                            'name' : 'orig',
                        },
                        ],
                        'layout':dict(
                            margin={'l':0, 'b':0, 't':0, 'r':0},
                            showlegend=False,
                            height=200,
                            # width=250,
                        )
                        },
                        className = class_name,
                    )
            )
    graphs.append(html.Div(inner.copy(), className='row'))
    return graphs
    
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')