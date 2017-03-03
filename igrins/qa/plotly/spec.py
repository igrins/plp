
# coding: utf-8

import numpy as np

def _make_frame(o):
    frame = {
        'name': str(o),
    }

    return frame

def _make_frame_xrange(o, wmin, wmax):
    "frame that only change the xrange"

    frame = {
        'name': str(o),
        'layout': {'xaxis': {'range':[wmin, wmax]}}
    }

    return frame

def _make_frame_yrange(o, wmin, wmax):
    "frame that only change the xrange"

    frame = {
        'name': str(o),
        'layout': {'yaxis': {'range':[wmin, wmax]}}
    }

    return frame


class Slider:
    def __init__(self, duration=300):
        self.duration = duration

    def get_sliders_dict(self):
        sliders_dict = {
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Order:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': self.duration, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': []
        }

        return sliders_dict

    def make_slider_step(self, o):
        slider_step = {'args': [
            [str(o)],
            {'frame': {'duration': self.duration, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': self.duration}}
         ],
         'label': o,
         'method': 'animate'}

        return slider_step


def figure_frame_range(orders, wvls, specs, reverse_order=True):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': False}
    }

    # setup layout
    figure['layout']['xaxis'] = {'title': 'Spectra'}
    figure['layout']['yaxis'] = {'title': 'ADU'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 400,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': 'all',
        'plotlycommand': 'animate',
        'values': [],
        'visible': True
    }

    # add buttons
    figure['layout']['updatemenus'] = [
        {'buttons': [
            {'label': 'Play',
             'args': [None, {'frame': {'duration': 600,
                                       'redraw': False},
                             'fromcurrent': True, 
                             'transition': {'duration': 300, 
                                            'easing': 'quadratic-in-out'}}],
             'method': 'animate'
            },
            {'label': 'Pause',
             'args': [[None], {'frame': {'duration': 0,
                                         'redraw': False},
                               'mode': 'immediate',
                               'transition': {'duration': 0}}],
             'method': 'animate'
            }
        ],
         'direction': 'left',
         'pad': {'r': 10, 't': 87},
         'showactive': False,
         'type': 'buttons',
         'x': 0.1,
         'xanchor': 'right',
         'y': 0,
         'yanchor': 'top'
        }
    ]


    # Slider object
    slider = Slider()

    # setup slider dict
    sliders_dict = slider.get_sliders_dict()


    ows = zip(orders, wvls, specs)
    if reverse_order:
        ows = ows[::-1]

    # add only first spec in the plot
    for o, w, s in ows[:]:
        data_dict = {
                'x': w,
                'y': s,
                'mode': 'lines',
                'name': "%d" % o,
                'line': {'simplify': False},
        }

        figure['data'].append(data_dict)
    
    # set up frames

    # first frame as an "all" frame
    if 1:
        o = "all"
        w0, w1 = ows[0][1], ows[-1][1]

        frame = _make_frame_xrange(o, w0[0], w1[-1])
        figure['frames'].append(frame)

        slider_step = slider.make_slider_step(o)
        sliders_dict['steps'].append(slider_step)


    for o, w, s in ows:
        frame = {}
        #w = np.arange(len(w))

        if 0:
            data_dict = {
                    'x': w,
                    'y': s,
                    'mode': 'lines',
                    'name': "%d" % o,
                    #'line': {'simplify': False},
            }

            frame['data'] = [data_dict]

        frame = _make_frame_xrange(o, w[0], w[-1])
        figure['frames'].append(frame)

        slider_step = slider.make_slider_step(o)
        sliders_dict['steps'].append(slider_step)
    
    figure['layout']['sliders'] = [sliders_dict]

    return figure


def figure_frame_data(orders, specs_list, 
                      name_list,
                      reverse_order=True, fix_yrange=True):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': False}
    }

    # setup layout
    figure['layout']['xaxis'] = {'title': 'Pixel'}
    figure['layout']['yaxis'] = {'title': 'ADU'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 400,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': 'all',
        'plotlycommand': 'animate',
        'values': [],
        'visible': True
    }

    # add buttons
    figure['layout']['updatemenus'] = [
        {'buttons': [
            {'label': 'Play',
             'args': [None, {'frame': {'duration': 600,
                                       'redraw': False},
                             'fromcurrent': True, 
                             'transition': {'duration': 300, 
                                            'easing': 'quadratic-in-out'}}],
             'method': 'animate'
            },
            {'label': 'Pause',
             'args': [[None], {'frame': {'duration': 0,
                                         'redraw': False},
                               'mode': 'immediate',
                               'transition': {'duration': 0}}],
             'method': 'animate'
            }
        ],
         'direction': 'left',
         'pad': {'r': 10, 't': 87},
         'showactive': False,
         'type': 'buttons',
         'x': 0.1,
         'xanchor': 'right',
         'y': 0,
         'yanchor': 'top'
        }
    ]


    # Slider object
    slider = Slider()

    # setup slider dict
    sliders_dict = slider.get_sliders_dict()

    os = zip(orders, zip(*specs_list))

    if reverse_order:
        os = os[::-1]

    # add only first spec in the plot
    for o, s_list in os[:1]:
        for s, name in zip(s_list, name_list):
            w = np.arange(len(s))
            data_dict = {
                    'x': w,
                    'y': np.nan_to_num(s),
                    'mode': 'line',
                    'name': name,
                    'line': {'simplify': False},
            }

            figure['data'].append(data_dict)

    
    # set up frames

    smax = np.nanpercentile(specs_list, 95)*1.1

    if fix_yrange:
        figure['layout']['yaxis'] = {'range':[0, smax]}


    for o, s_list in os[:]:
        if fix_yrange:
            frame = _make_frame_yrange(o, 0, smax)
        else:
            frame = _make_frame(o)

        frame['data'] = []

        for s, name in zip(s_list, name_list):
            w = np.arange(len(s))

            data_dict = {
                    'x': w,
                    'y': np.nan_to_num(s),
                    'mode': 'line',
                    'name': name,
                    'line': {'simplify': False},
            }

            frame['data'].append(data_dict)


        figure['frames'].append(frame)

        slider_step = slider.make_slider_step(o)
        sliders_dict['steps'].append(slider_step)
    
    figure['layout']['sliders'] = [sliders_dict]

    return figure


if __name__ == "__main__":

    import plotly
    plotly.offline.init_notebook_mode(connected=False)



    import astropy.io.fits as pyfits
    import numpy as np
    f = pyfits.open("SDCK_20150305_0044.spec.fits")
    specs = f[0].data
    wvls = f[1].data
    orders = np.arange(len(specs))

    figure = spec_figure(orders, wvls, specs)

    import plotly_offline_jjlee
    plotly_offline_jjlee.iplot(figure)


    if 0:
        from requests.compat import json as _json
        from plotly import utils
        jdata = _json.dumps(figure, cls=utils.PlotlyJSONEncoder)

        import json
        k = json.loads(jdata)

        import msgpack
        d = msgpack.packb([1,2,3])
        import base64
        ds = base64.encodestring(d)

        import bson
        b = bson.dumps(k)
        bs = base64.encodestring(b)


        print len(jdata), len(ds), len(bs)
        print len(jdata), len(d), len(b)


