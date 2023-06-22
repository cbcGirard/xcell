#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptation parameters
================================

Illustrates how adaptation parameters affect the generated mesh

"""

import Common
import numpy as np
import xcell
import matplotlib.pyplot as plt

# swept='density'
# swept = 'depth'
dmin = 2
dmax = 6

study, setup = Common.makeSynthStudy('adaptationDemos')
plotprefs={'colorbar':False,
           'barRatio':[7,1],
           'labelAxes':False}

sweepval = 1-abs(np.linspace(-1, 1, 20))

for swept in ['density','depth']:

    if swept == 'density':
        vrange = 0.5*sweepval
    elif swept == 'depth':
        # vrange=dmin+np.array((dmax-dmin)*sweepval,dtype=int)
        vrange = np.concatenate(
            (np.arange(dmin, dmax+1), np.arange(dmax-1, dmin-1, -1)))
    tvec = np.linspace(0, 1, vrange.shape[0])

    tdata = {
        'x': tvec,
        'y': vrange,
        'ylabel': swept,
        'unit': '',
        'style': 'dot'
    }

    with plt.rc_context({
            'figure.figsize':[4.5,3.75],
            'font.size':10,
            'figure.dpi':144
            }):

        img = xcell.visualizers.SingleSlice(None, study, timevec=tvec, tdata=tdata, prefs=plotprefs)
        # aa=img.axes[0]
        # aa.set_xticks([])
        # aa.set_yticks([])
        img.axes[2].set_xticks([])

        # data={
        #       'mesh':[],
        #       'depth':[],
        #       'density':[]}

        lastdepth = -1
        for val in vrange:

            if swept == 'density':
                density = val
                maxdepth = dmax
            elif swept == 'depth':
                maxdepth = val
                density = .2

            metric = xcell.generalMetric

            setup.makeAdaptiveGrid(np.zeros((1, 3)),  maxdepth=np.array(maxdepth,ndmin=1), minl0Function=metric,  coefs=np.ones(1)*2**(-maxdepth*density))
            setup.finalizeMesh()

            img.addSimulationData(setup, append=True)


        ani = img.animateStudy(fname=swept, fps=5.)
# sphinx_gallery_thumbnail_number = 2