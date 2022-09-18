#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:53:44 2022

@author: benoit
"""

import xcell as xc
import Applications.Common as com
import numpy as np
import matplotlib.pyplot as plt


ring=com.Ring(N=5,dendSegs=3,r=175)
maxdepth=6
density=0.1

_,_,pts,_=xc.nrnutil.getNeuronGeometry()
coords=np.array(pts)

# coords[:,-1]=0

xmax=1.5*np.max(np.abs(coords))
bnds=xmax*np.array([-1,1,-1,1])


setup=xc.Simulation('',xmax*np.array([-1,-1,-1,1,1,1]))

scale=np.ones(coords.shape[0])*2**(-maxdepth*density)
setup.makeAdaptiveGrid(coords, maxdepth, xc.generalMetric,      coefs=scale)
setup.finalizeMesh()


# xc.visualizers.formatXYAxis(ax,bnds)
view=xc.visualizers.SliceViewer(None, setup)
xc.nrnutil.showCellGeo(view.ax)

