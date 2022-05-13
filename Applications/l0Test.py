#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:59:01 2021

@author: benoit
"""

import xcell
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import resource 

xmax=1e-4

maxDepth=8

sigma=np.ones(3,dtype=np.float64)
vMode=False
elType='Admittance'
fname='l0_current.csv'

resultpath="Results/cube/errEst/"

def runL0Grid(maxDepth,coef,xmax,showGraphs,vMode,logTimes):
    bbox=np.concatenate((-xmax*np.ones(3), xmax*np.ones(3)))
    
    otree=xcell.Octree(bbox,maxDepth=maxDepth)
    setup=xcell.Simulation(resultpath)
    setup.mesh.extents=2*xmax*np.ones(3)
    setup.mesh.elementType=elType
    
    
    def inverseSquare(coord):
        return 1/np.dot(coord,coord)
    
    
    @nb.njit
    def toR(coord,coef=coef):
        r=np.linalg.norm(coord)
        # val=coef*np.linalg.norm(coord)
        # val=(coef*r**2)**(1/3) #current continuity
        val=(coef*r**4)**(1/3) #dirichlet energy continutity
        if val<1e-6:
            val=1e-6
        return val
    
    metric=toR
    
    setup.startTiming("Make elements")
    otree.refineByMetric(metric)

    
    setup.mesh=otree.makeMesh(setup.mesh)
    coords=setup.mesh.nodeCoords
    
    setup.logTime()
    numEl=len(setup.mesh.elements)
    
    sourceIndex=otree.coord2Index(np.zeros(3))
    
    if vMode:
        srcMag=1
    
        # setup.addVoltageSource(srcMag,index=sourceIndex)
        srcType='Voltage'
    else:
        srcMag=1e-6
        setup.addCurrentSource(srcMag,index=sourceIndex)
        srcType='Current'
    
    # ground boundary nodes, set v sources
    for ii in nb.prange(coords.shape[0]):
        pt=np.abs(coords[ii])
        rpt=np.array(np.linalg.norm(pt))
        if np.any(pt==xmax) or (rpt<=1e-6):
            # setup.addVoltageSource(0,index=ii)
            
            vpt=xcell.analyticVsrc(np.zeros(3), srcMag, rpt,srcType=srcType)
            setup.addVoltageSource(vpt.squeeze(),index=ii)
        
    
    
    v=setup.solve()
    # v=setup.iterativeSolve(None,1e-9)
    setup.getMemUsage(True)
    FVU=setup.calculateErrors(srcMag,srcType,showPlots=showGraphs)

    
    
    
    if logTimes:
        setup.logAsTableEntry(resultpath+fname, FVU,
                              ['vSource'],
                              [v[sourceIndex]])
        
        
    times=np.array([l.duration for l in setup.stepLogs])
    print('\n\n\tTotal Time: %g' % np.sum(times))
    print(numEl**(1/3))
# for co in np.flip(np.arange(1,10)/10):
#     # print('\n\nDepth: %d'%md)
#     print("coef: %g"%co)
#     for xx in np.logspace(-5,-3,10):
#         runL0Grid(12,co, xx, showGraphs=False, vMode=False, logTimes=True)

runL0Grid(5, 1e4,1e-4, showGraphs=True, vMode=False, logTimes=False)
