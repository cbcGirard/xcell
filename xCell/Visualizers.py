#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:15:04 2021
Visualization routines for meshes
@author: benoit
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import matplotlib as mpl
import numpy as np


def showEdges(axis,coords, edgeIndices, edgeVals=None,colorbar=True):
    edgePts=[[coords[a,:],coords[b,:]] for a,b in edgeIndices]
    
    if edgeVals is not None:
        (cMap, cNorm) = getCmap(edgeVals)
        gColor = cMap(cNorm(edgeVals))
        alpha=1.
    else:
        # gColor=np.repeat([0,0,0,1], edgeIndices.shape[0])
        gColor=(0.,0.,0.)
        alpha=0.05
    
    gCollection=p3d.art3d.Line3DCollection(edgePts,colors=gColor,alpha=alpha)
    
    axis.add_collection(gCollection)

    if colorbar:
        axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))
    
    
def showNodes(axis, coords, nodeVals):
    
    (cMap, cNorm) = getCmap(nodeVals)
    vColors = cMap(cNorm(nodeVals))
    x,y,z=np.hsplit(coords,3)
    
    axis.scatter3D(x,y,z, c=vColors)
    axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))


def getCmap(vals):
    if min(vals) < 0:
        cMap = mpl.cm.seismic
        cNorm = mpl.colors.CenteredNorm()
    else:
        cMap = mpl.cm.plasma
        cNorm = mpl.colors.Normalize()
    return (cMap, cNorm)


def new3dPlot(boundingBox):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    fig=plt.figure()
    axis=fig.add_subplot(projection='3d')
    origin=boundingBox[:3]
    span=boundingBox[3:]-origin
    center=0.5*(origin+boundingBox[3:])
    
    max_range=max(span)    
    
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*center[0]
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*center[1]
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*center[2]
    for xb, yb, zb in zip(Xb, Yb, Zb):
       axis.plot([xb], [yb], [zb], color=[1,1,1,0])

    return axis

def equalizeScale(ax):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    xy=ax.xy_dataLim
    zz=ax.zz_dataLim
    
    max_range = np.array([xy.xmax-xy.xmin,xy.ymax-xy.ymin, zz.max.max()-zz.min.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xy.xmax+xy.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(xy.ymax+xy.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zz.max.max()+zz.min.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], color=[1,1,1,0])

# def showNodeValues(axis,coords, nodeVals):
    
    