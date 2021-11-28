#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:56:54 2021

@author: benoit
"""

import numpy as np
import numba as nb
import xCell

import dill

studyPath='Results/studyTst'

maxdepth=10
xmax=1e-4
sigma=1


vMode=False
showGraphs=False


vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)


if vMode:
    study.vSourceCoords=np.zeros()
    study.vSourceVals=vsrc
else:
    study.iSourceCoords=np.zeros(3)
    study.iSourceVlas=isrc



# for l0Param in np.logspace(0,-1,10):
# # for l0Param in [0.5]:

#     def metric(coord,l0Param=l0Param):
#         r=np.linalg.norm(coord)
#         val=l0Param*r
#         if val<1e-6:
#             val=1e-6
#         return val
    
#     setup=study.newSimulation()
#     setup.mesh.elementType='Admittance'
#     otree=xCell.Octree(bbox,maxDepth=maxdepth)

#     setup.startTiming("Make elements")
#     otree.refineByMetric(metric)
#     setup.mesh=otree.makeMesh(setup.mesh)
#     coords=setup.mesh.nodeCoords
    
    
        
#     setup.logTime()
#     numEl=len(setup.mesh.elements)
    
#     sourceIndex=otree.coord2Index(np.zeros(3))
    
#     if vMode:
#         srcMag=1
    
#         # setup.addVoltageSource(srcMag,index=sourceIndex)
#         srcType='Voltage'
#     else:
#         srcMag=1e-6
#         setup.addCurrentSource(srcMag,index=sourceIndex)
#         srcType='Current'
    
#     # ground boundary nodes, set v sources
#     for ii in nb.prange(coords.shape[0]):
#         pt=np.abs(coords[ii])
#         rpt=np.array(np.linalg.norm(pt))
#         if np.any(pt==xmax) or (rpt<=1e-6):
#             # setup.addVoltageSource(0,index=ii)
            
#             vpt=xCell.analyticVsrc(np.zeros(3), srcMag, rpt,srcType=srcType)
#             setup.addVoltageSource(vpt.squeeze(),index=ii)
        
    
    
#     v=setup.solve()
#     # v=setup.iterativeSolve(None,1e-9)
#     setup.getMemUsage(True)
#     FVU=setup.calculateErrors(srcMag,srcType,showPlots=showGraphs)
    
#     study.saveData(setup)


# ani=study.animatePlot(xCell.error2d,'ani')
ani=study.animatePlot(xCell.)