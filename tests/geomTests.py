#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:41:54 2022

@author: benoit
"""

import Geometry as geo
import Visualizers as viz
import numpy as np

def showBound(obj):
    xx=np.linspace(-1,1,21)
    X,Y,Z=np.meshgrid(xx,xx,xx)
    pts=np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()
    isin=obj.isInside(pts)

    ax=viz.new3dPlot()
    viz.showNodes3d(ax, pts, isin)





origin=np.zeros(3)
axis=np.random.rand(3)
length=1.
radius=0.5


cyl=geo.Cylinder(origin, radius, length, axis)

disk=geo.Disk(origin,radius,axis)

showBound(disk)
showBound(cyl)
