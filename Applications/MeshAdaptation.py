#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:25:55 2022

Illusrates how adaptation parameters affect the generated mesh


@author: benoit
"""

import Common
import numpy as np
import xcell
import matplotlib.pyplot as plt


swept = 'depth'
dmin = 2
dmax = 6

study, setup = Common.makeSynthStudy('adaptationDemos')


sweepval = 1-abs(np.linspace(-1, 1, 20))

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
    'unit': None,
    'style': 'dot'
}


img = xcell.visualizers.SingleSlice(None, study, timevec=tvec, tdata=tdata)


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

        # if lastdepth==maxdepth:
        #     continue
        # else:
        #     lastdepth=maxdepth

    metric = xcell.generalMetric

    setup.makeAdaptiveGrid(np.zeros((1, 3)),  maxdepth=np.array(maxdepth,ndmin=1), minl0Function=metric,  coefs=np.ones(1)*2**(-maxdepth*density))
    setup.finalizeMesh()
    # _,_,edgePts=setup.getElementsInPlane()

    img.addSimulationData(setup, append=True)

    # ax=xcell.visualizers.showMesh(setup)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ghost=(.0, .0, .0, 0.0)
    # ax.xaxis.set_pane_color(ghost)
    # ax.yaxis.set_pane_color(ghost)
    # ax.zaxis.set_pane_color(ghost)


ani = img.animateStudy(fname=swept, fps=5.)

lcopy=img.makeLightCopy()

lani=lcopy.animateStudy(fps=5)