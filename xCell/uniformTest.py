#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:35:30 2021

@author: benoit
"""

import numpy as np
import numba as nb
import scipy as scp

from numba.types import int64, float64


import xCell
import matplotlib.pyplot as plt


showGraphs=False
fname="Results/cube/table0-5.csv"

xmax=.5
#must be even
nDiv=10

# elType='FEM'
elType='Admittance'

def runUniformGrid(nDiv,elType,xmax,showGraphs):
    
    
    
    iSrcMag=1
    
    sigma=np.array([1.,1.,1.])
    
    
    boundingBox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
    
    xx=np.linspace(-xmax,xmax,nDiv+1)
    XX,YY,ZZ=np.meshgrid(xx,xx,xx)
    
    
    coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()
    
    setup=xCell.Simulation()
    
    setup.mesh.nodeCoords=coords
    setup.mesh.extents=2*xmax*np.ones(3)
    setup.mesh.elementType=elType
    
    # place unit source at center 
    sourceIndex=np.floor_divide(coords.shape[0],2)
    setup.addCurrentSource(1,index=sourceIndex)
    
    # ground boundary nodes
    for ii in nb.prange(coords.shape[0]):
        pt=np.abs(coords[ii])
        if np.any(pt==xmax):
            setup.addVoltageSource(0,index=ii)
    
    
    
    elExtents=setup.mesh.extents/nDiv
    
    elOffsets=np.array([1,nDiv+1,(nDiv+1)**2])
    nodeOffsets=np.array([np.dot(xCell.toBitArray(i),elOffsets) for i in range(8)])
    
    
    setup.startTiming("Make elements")
    for zz in range(nDiv):
        for yy in range(nDiv):
            for xx in range(nDiv):
                elOriginNode=xx+yy*(nDiv+1)+zz*(nDiv+1)**2
                origin=coords[elOriginNode]
                elementNodes=elOriginNode+nodeOffsets
                
                setup.mesh.addElement(origin, elExtents, sigma,elementNodes)
                
    setup.logTime()            
    
    print('%d nodes, %d elements' % (coords.shape[0],len(setup.mesh.elements)))
                
    # edges,conductances=setup.mesh.getConductances()
    
    
    # ax=xCell.new3dPlot(boundingBox)
    # xCell.showEdges(ax, coords, edges, conductances)
    
    v=setup.solve()
    
    
    
    #TODO: reorder r to mask singularity
    r=np.linalg.norm(coords,axis=1)
    analytic=np.array([iSrcMag/(4*np.pi*d) for d in r])
    
    err=v-analytic
    err[sourceIndex]=0 #mask singularity at 0 radius
    
    errRMS=np.linalg.norm(err)/len(err)
    errRel=err/analytic
    
    if showGraphs:
    
        axSol=xCell.new3dPlot(boundingBox)
        edges,conds=setup.mesh.getConductances()
        xCell.showEdges(axSol, coords, edges, colorbar=False)

        xCell.showNodes(axSol,coords,v)
        plt.title('Approximate solution')
        
        
        
        axErr=xCell.new3dPlot(boundingBox)
        xCell.showNodes(axErr, coords, err)
        plt.title('Solution Error\n RMS %.2g' % errRMS)
        
        nNodes=coords.shape[0]
        nElems=len(setup.mesh.elements)
        
        
        fig2d, axes=plt.subplots(2,1)
        ax2dA,ax2dB=axes
        ax2dA.scatter(r[1:],v[1:],label='Approximation')
        ax2dA.scatter(r[1:],analytic[1:], label='Analytical')
        ax2dA.legend()
        ax2dA.set_title('%d nodes, %d %s elements\nRMS error %g'%(nNodes,nElems,elType,errRMS))
        ax2dA.set_xlabel('Distance from source [m]')
        ax2dA.set_ylabel('Voltage [V]')
        
        ax2dB.scatter(r[1:],errRel[1:],c='r',label='Relative')
        ax2dB.set_ylabel('Relative error')
        ax2dC=ax2dB.twinx()
        ax2dC.scatter(r[1:],err[1:],c='orange',label='Absolute')
        ax2dC.set_ylabel('Absolute error [V]')
        plt.legend()
    
    else:
        setup.logAsTableEntry(fname, errRMS)
        times=np.array([l.duration for l in setup.stepLogs])
        print('\n\n\tTotal Time: %g' % np.sum(times))

# runUniformGrid(4, elType, xmax, True)

for ii in range(30,100,5):
    runUniformGrid(ii*2, elType, xmax,showGraphs=False)