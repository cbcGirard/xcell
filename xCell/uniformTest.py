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


showGraphs=True
fname="Results/cube/voltageMode.csv"

xmax=1e-5
#must be even
nDiv=30

# elType='FEM'
elType='Admittance'

vMode=True
def runUniformGrid(nDiv,elType,xmax,showGraphs,vMode=False,logTimes=False):
    
    plt.close('all')
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
    
    if vMode:
        srcMag=1

        setup.addVoltageSource(srcMag,index=sourceIndex)
        srcType='Voltage'
    else:
        srcMag=1e-9
        setup.addCurrentSource(srcMag,index=sourceIndex)
        srcType='Current'
    
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
    
    
    
    r=np.linalg.norm(coords,axis=1)

    analytic=xCell.analyticVsrc(np.zeros(3), srcMag,r,srcType=srcType)
    
    
    
    rDense=np.linspace(0,max(r),100)
    analyticDense=xCell.analyticVsrc(np.zeros(3), srcMag, rDense,srcType=srcType)
    # analytic=np.array([srcMag/(4*np.pi*d) for d in r])
    
    err=v-analytic
    
    errRMS=np.linalg.norm(err)/len(err)
    errRel=err/analytic
    
    if showGraphs:
    
        
        xCell.showRawSlice(v, nDiv)
        plt.title('Approximate solution [V]')
        
        plt.figure()
        xCell.showRawSlice(err, nDiv)
        plt.title('Error')
        # axSol=xCell.new3dPlot(boundingBox)
        # edges,conds=setup.mesh.getConductances()
        # xCell.showEdges(axSol, coords, edges, colorbar=False)
    
        # xCell.showNodes(axSol,coords,v)
        # plt.title('Approximate solution')
        
        axCurrent=xCell.new3dPlot(boundingBox)
        
        mids=[]
        iVecs=[]
        
        for el in setup.mesh.elements:
            vElem=v[el.globalNodeIndices]
            interp=xCell.getElementInterpolant(el, vElem)
            mid=el.getMidpoint()
            
            ivec=xCell.getCurrentVector(interp, mid)
            
            mids.append(mid)
            iVecs.append(ivec)
            
        
        xCell.showCurrentVecs(axCurrent, np.array(mids), np.array(iVecs))
        # axErr=xCell.new3dPlot(boundingBox)
        # xCell.showNodes(axErr, coords, err)
        # plt.title('Solution Error\n RMS %.2g' % errRMS)
        
        nNodes=coords.shape[0]
        nElems=len(setup.mesh.elements)
        
        
        fig2d, axes=plt.subplots(2,1)
        ax2dA,ax2dB=axes
        ax2dA.scatter(r,v,c='r',label='Approximation')
        ax2dA.plot(rDense,analyticDense, label='Analytical')
        ax2dA.legend()
        ax2dA.set_title('%d nodes, %d %s elements\nRMS error %g'%(nNodes,nElems,elType,errRMS))
        ax2dA.set_xlabel('Distance from source [m]')
        ax2dA.set_ylabel('Voltage [V]')
        
        ax2dB.scatter(r,err,c='r')
        ax2dB.set_ylabel('Absolute error [V]')
        plt.tight_layout()
        # ax2dC=ax2dB.twinx()
        # ax2dC.scatter(r[1:],err[1:],c='orange',label='Absolute')
        # ax2dC.set_ylabel('Absolute error [V]')
        # plt.legend()
    
    if logTimes:
        #hack for logging dimensions
        setup.elementType=xmax
        setup.logAsTableEntry(fname, errRMS)
        times=np.array([l.duration for l in setup.stepLogs])
        print('\n\n\tTotal Time: %g' % np.sum(times))
       
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        


# runUniformGrid(nDiv, elType, xmax, showGraphs=True,vMode=True,logTimes=False)
runUniformGrid(64, elType, 3e-5, showGraphs=True,vMode=True,logTimes=False)
# for nd in range(4,40,4):
#     for xx in np.logspace(-5,0):
#         runUniformGrid(nd, elType, xx,showGraphs=False,vMode=True,logTimes=True)