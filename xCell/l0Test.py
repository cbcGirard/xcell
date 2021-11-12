#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:59:01 2021

@author: benoit
"""

import xCell
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import resource 

xmax=1e-3

maxDepth=3

sigma=np.ones(3,dtype=np.float64)
vMode=False
elType='Admittance'
fname='Results/cube/l0_currentMode.csv'

def runL0Grid(maxDepth,xmax,showGraphs,vMode,logTimes):
    bbox=np.concatenate((-xmax*np.ones(3), xmax*np.ones(3)))
    
    otree=xCell.Octree(bbox,maxDepth=maxDepth)
    setup=xCell.Simulation()
    setup.mesh.extents=2*xmax*np.ones(3)
    setup.mesh.elementType=elType
    
    
    def inverseSquare(coord):
        return 1/np.dot(coord,coord)
    
    def toR(coord):
        return 0.5*np.linalg.norm(coord)
    
    
    setup.startTiming("Make elements")
    otree.refineByMetric(toR)
    # otree.printStructure()
    numEl=otree.countElements()
    octs=otree.tree.getTerminalOctants()
    
    
    coords=otree.getCoordsRecursively()
    setup.mesh.nodeCoords=coords
    
    
    # ax=xCell.new3dPlot(bbox)
    
    # X,Y,Z=np.hsplit(coords,3)
    # ax.scatter3D(X,Y,Z)
    
    
    for o in octs:
        # o.calcGlobalIndices(bbox, maxDepth)
        ocoords=o.getOwnCoords()
        # oNdx=np.array([otree.coord2Index(c) for c in ocoords])
        # globMask=np.isin(indices,oNdx,assume_unique=True)
        # globndx=indices[globMask]
        # globs=[np.]
        setup.mesh.addElement(o.origin, o.span, sigma, o.globalNodes)
        
    setup.logTime()
    
    sourceIndex=otree.coord2Index(np.zeros(3))
    
    if vMode:
        srcMag=1
    
        setup.addVoltageSource(srcMag,index=sourceIndex)
        srcType='Voltage'
    else:
        srcMag=1e-6
        setup.addCurrentSource(srcMag,index=sourceIndex)
        srcType='Current'
    
    # ground boundary nodes
    for ii in nb.prange(coords.shape[0]):
        pt=np.abs(coords[ii])
        if np.any(pt==xmax):
            setup.addVoltageSource(0,index=ii)
        
        
        
    # Show conductances along edges
    edges,conductances=setup.mesh.getConductances()
    
    # ax=xCell.new3dPlot(bbox)
    # xCell.showEdges(ax, coords, edges, conductances)
    
    
    v=setup.solve()
    
    
    r=np.linalg.norm(coords,axis=1)
        
    #redact singularity
    rest=r>0
    # r=r[rest]
    # v=v[rest]
    
    filtR=r[rest]
    filtCoords=coords[rest]
    filtV=v[rest]
    
    # filtEdges=[eg for eg in edges if not any(eg==sourceIndex())]
    
    analytic=xCell.analyticVsrc(np.zeros(3), srcMag,r,srcType=srcType)
    
    fit=np.polyfit(analytic, v, 1)
    
    
    rDense=np.linspace(min(r[r>0]),max(r),100)
    # rDense=r[r>0]
    analyticDense=xCell.analyticVsrc(np.zeros(3), srcMag, rDense,srcType=srcType)
    # analytic=np.array([srcMag/(4*np.pi*d) for d in r])
    
    
    err=v-analytic
    err[sourceIndex]=0
    relErr=err/analytic
    
    errRMS=np.linalg.norm(err[rest])/len(err[rest])
    
    
    if showGraphs:
        ##Plots
        dX=2**maxDepth +1
        plt.figure()
        xCell.showSlice(coords, v,dX,edges=edges)
        plt.title('Estimated value')
        plt.tight_layout()
        
        plt.figure()
        xCell.showSlice(coords, relErr, dX,edges=edges)
        plt.title('Relative error')
        plt.tight_layout()
        
        nNodes=coords.shape[0]
        nElems=len(setup.mesh.elements)
        
        
        fig2d, axes=plt.subplots(2,1)
        ax2dA,ax2dB=axes
        ax2dA.scatter(r[rest],v[rest],c='r',label='Approximation')
        ax2dA.plot(rDense,analyticDense, label='Analytical')
        ax2dA.legend()
        ax2dA.set_title('%d nodes, %d %s elements\nRMS error %g'%(nNodes,nElems,elType,errRMS))
        ax2dA.set_xlabel('Distance from source [m]')
        ax2dA.set_ylabel('Voltage [V]')
        
        ax2dB.scatter(r[rest],err[rest],c='r',label='Absolute')
        ax2dB.set_ylabel('Absolute error [V]')
        plt.tight_layout()
        ax2dC=ax2dB.twinx()
        ax2dC.scatter(r[rest],relErr[rest],c='orange',label='Relative')
        ax2dC.set_ylabel('Relative error')
        plt.legend()
        plt.tight_layout()
        
        plt.figure()
        ref=np.linspace(0,max(v))
        plt.scatter(analytic,v)
        plt.xlabel('Analytical')
        plt.ylabel('Approximation')
        
        pf=np.poly1d(fit)
        plt.plot(ref,pf(ref))
        plt.title('%gx%+g'%(fit[0],fit[1]))
    
    
    if logTimes:
        #hack for logging dimensions
        setup.logAsTableEntry(fname, errRMS,
                              ['max error',
                               'slope',
                               'offset',
                               'RMS relative',
                               'Max relative'],
                              [max(abs(err)),
                               fit[0],
                               fit[1],
                              np.linalg.norm(relErr),
                              max(abs(relErr))])
        
        
    times=np.array([l.duration for l in setup.stepLogs])
    print('\n\n\tTotal Time: %g' % np.sum(times))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
# for md in range(2,20):
#     print('\n\nDepth: %d'%md)
#     for xx in np.logspace(-4,0,20):
#         runL0Grid(md, xx, showGraphs=False, vMode=vMode, logTimes=True)

runL0Grid(7, 1e-4, showGraphs=True, vMode=False, logTimes=False)