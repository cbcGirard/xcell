#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:29:09 2022

@author: benoit
"""

import xCell
import numpy as np
import Common
import matplotlib.pyplot as plt

maxdepth=4

study,setup=Common.makeSynthStudy('Singularity')

plotter=xCell.Visualizers.ErrorGraph(plt.figure(), study)

def boundaryFun(coord):
    r=np.linalg.norm(coord)
    val=1e-6/(r*np.pi*4)
    return val


etots=[]
esrc=[]
nInSrc=[]
rclosest=[]
for l0min in np.logspace(-5, -7):
    maxdepth=np.floor(np.log2(1e-4/l0min))+2
    xmax=l0min*2**maxdepth
    study,setup=Common.makeSynthStudy('Singularity',xmax=xmax)

    metric=xCell.makeExplicitLinearMetric(maxdepth, 0.2)
    # metric=xCell.makeBoundedLinearMetric(l0min, 
    #                                       xmax/4, 
    #                                       xmax)
    setup.makeAdaptiveGrid(metric, maxdepth)
    
    setup.finalizeMesh()
    setup.setBoundaryNodes(boundaryFun)
    
    v=setup.iterativeSolve()
    # setup.applyTransforms()
    
    setup.getMemUsage(True)
    setup.printTotalTime()
    
    plotter.addSimulationData(setup)
    emetric,evec,_,sortr=setup.calculateErrors()
    etots.append(emetric)
    esrc.append(evec[sortr][0])
    nInSrc.append(sum(setup.nodeRoleTable==2))
    
    r=np.linalg.norm(setup.mesh.nodeCoords,axis=1)
    
    # rclose=min(r[setup.nodeRoleTable!=2])
    rclose=min(r[r!=0])

    rclosest.append(rclose)
    
plotter.dataSets=[plotter.dataSets]
    
ani=plotter.animateStudy(fps=10)

    
f2,axes=plt.subplots(3,sharex=True,gridspec_kw={'height_ratios':[4,4,2]})
[ax.grid(True) for ax in axes]
ax=axes[0]
a3=axes[2]
# a3.sharex(ax)

ax.set_xscale('log')

# totcol='tab:orange'
# pkcol='tab:red'
totcol='k'
pkcol='k'
ax.plot(rclosest,etots,color=totcol)
ax.set_ylabel('Total error',color=totcol)

# ax2=ax.twinx()
ax2=axes[1]
# ax2.sharex(ax)
ax2.plot(rclosest,esrc,color=pkcol)
ax2.set_ylabel('Error at source [V]',color=pkcol)


a3.plot(rclosest,nInSrc,color='k')
a3.set_ylabel('Points in source')
a3.set_xlabel('Closest node to origin [m]')
a3.set_yscale('log')