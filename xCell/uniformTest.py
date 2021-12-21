#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:35:30 2021

@author: benoit
"""

import numpy as np
import numba as nb
import scipy as scp
import resource

from numba.types import int64, float64


import xCell
import matplotlib.pyplot as plt


studyPath='Results/studyTst'

maxdepth=10
xmax=1e-4
sigma=np.ones(3)
rElec=1e-6


vMode=False
showGraphs=False
generate=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)





# def runUniformGrid(nDiv,elType,xmax,showGraphs,vMode=False,logTimes=False):
nDiv=6

# plt.close('all')
# sigma=np.array([1.,1.,1.])


# boundingBox=np.append(-xmax*np.ones(3),xmax*np.ones(3))

# xx=np.linspace(-xmax,xmax,nDiv+1)
# XX,YY,ZZ=np.meshgrid(xx,xx,xx)


# coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()
# r=np.linalg.norm(coords,axis=1)
# setup=xCell.simulation(resultpath)
setup=study.newSimulation()

# setup.mesh.nodeCoords=coords
# setup.mesh.extents=2*xmax*np.ones(3)
# setup.mesh.elementType=elType
setup.makeUniformGrid(nDiv)
coords=setup.mesh.nodeCoords

# place unit source at center 
sourceIndex=np.floor_divide(coords.shape[0],2)

if vMode:
    setup.addVoltageSource(1,np.zeros(3),rElec)
    srcMag=1.
    srcType='Voltage'
else:
    srcMag=4*np.pi*sigma[0]*rElec
    setup.addCurrentSource(srcMag,np.zeros(3),rElec)
    # setup.addCurrentSource(srcMag,index=sourceIndex)
    srcType='Current'

# ground boundary nodes
# for ii in nb.prange(coords.shape[0]):
#     pt=np.abs(coords[ii])
#     rpt=np.array(np.linalg.norm(pt))
#     if np.any(pt==xmax) or (rpt<=1e-6):
#         # setup.addVoltageSource(0,index=ii)
        
#         vpt=xCell.analyticVsrc(np.zeros(3), srcMag, rpt,srcType=srcType)
#         setup.addVoltageSource(vpt.squeeze(),index=ii)

def boundaryFun(coord):
    r=np.linalg.norm(coord)
    return rElec/(r*np.pi*4)

setup.finalizeMesh()
setup.setBoundaryNodes(boundaryFun)

# elOffsets=np.array([1,nDiv+1,(nDiv+1)**2])
# nodeOffsets=np.array([np.dot(xCell.toBitArray(i),elOffsets) for i in range(8)])


# setup.startTiming("Make elements")
# for zz in range(nDiv):
#     for yy in range(nDiv):
#         for xx in range(nDiv):
#             elOriginNode=xx+yy*(nDiv+1)+zz*(nDiv+1)**2
#             origin=coords[elOriginNode]
#             elementNodes=elOriginNode+nodeOffsets
            
#             setup.mesh.addElement(origin, elExtents, sigma,elementNodes)
            
# setup.logTime()            

print('%d nodes, %d elements' % (coords.shape[0],len(setup.mesh.elements)))
            
edges,conductances=setup.mesh.getConductances()


# ax=xCell.new3dPlot(boundingBox)
# xCell.showEdges(ax, coords, edges, conductances)

# v=setup.solve()
v=setup.iterativeSolve(None,1e-9)

setup.getMemUsage(True)

err,FVU=setup.calculateErrors()#srcMag,srcType,showPlots=showGraphs)

# xCell.error2d(plt.figure(),setup)
xCell.centerSlice(plt.figure(),setup)

# if logTimes:
#     setup.logAsTableEntry(resultpath+fname, FVU,
#                           ['vSource'],
#                           [v[sourceIndex]])
    
    
times=np.array([l.duration for l in setup.stepLogs])
print('\n\n\tTotal Time: %g' % np.sum(times))
        


# runUniformGrid(nDiv, elType, xmax, showGraphs=True,vMode=True,logTimes=False)
# runUniformGrid(18, 1e-3, xmax=xmax,showGraphs=True,vMode=False,logTimes=False)

