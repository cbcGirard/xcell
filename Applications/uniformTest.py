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


import xcell
import matplotlib.pyplot as plt


studyPath='Results/studyTst'

nDiv=6
maxdepth=10
xmax=3e-6

sigma=np.ones(3)
rElec=1e-6


vMode=False
showGraphs=False
generate=False


def metric(coord):
    r=np.linalg.norm(coord)
    l0target=0.2*r
    
    #avoid subdivision inside source
    if (r+l0target)<rElec:
        l0target=rElec
    return l0target

def boundaryFun(coord):
    r=np.linalg.norm(coord)
    return rElec/(r*np.pi*4)

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xcell.SimStudy(studyPath,bbox)


# setup=study.newSimulation()
# if vMode:
#     setup.addVoltageSource(1,np.zeros(3),rElec)
#     srcMag=1.
#     srcType='Voltage'
# else:
#     srcMag=4*np.pi*sigma[0]*rElec
#     setup.addCurrentSource(srcMag,np.zeros(3),rElec)
#     # setup.addCurrentSource(srcMag,index=sourceIndex)
#     srcType='Current'
    
# # setup.makeUniformGrid(nDiv)
# setup.makeAdaptiveGrid(metric, maxdepth)



# setup.finalizeMesh()
# setup.setBoundaryNodes(boundaryFun)
# coords=setup.mesh.nodeCoords    

# print('%d nodes, %d elements' % (coords.shape[0],len(setup.mesh.elements)))
          

    

# # edges,conductances=setup.mesh.getConductances()
# # ax=xcell.new3dPlot(boundingBox)
# # xcell.showEdges(ax, coords, edges, conductances)

# # v=setup.solve()
# v=setup.iterativeSolve(None,1e-9)

# setup.getMemUsage(True)

# err,FVU=setup.calculateErrors()#srcMag,srcType,showPlots=showGraphs)
# study.saveData(setup)

setup=study.loadData('sim0')

# xcell.error2d(plt.figure(),setup)
xcell.centerSlice(plt.figure(),setup)

# if logTimes:
#     setup.logAsTableEntry(resultpath+fname, FVU,
#                           ['vSource'],
#                           [v[sourceIndex]])
    
    
times=np.array([l.duration for l in setup.stepLogs])
print('\n\n\tTotal Time: %g' % np.sum(times))
        


# runUniformGrid(nDiv, elType, xmax, showGraphs=True,vMode=True,logTimes=False)
# runUniformGrid(18, 1e-3, xmax=xmax,showGraphs=True,vMode=False,logTimes=False)

