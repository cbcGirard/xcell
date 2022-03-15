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
import pickle

maxdepth=4

fname='Singularity_sweep'
animate=False

rElec=1e-6
x0max=rElec*200
study0,_=Common.makeSynthStudy(fname,
                                  rElec=rElec,
                                  xmax=x0max)

elRs=1e-6*np.array([1,2,5])
x0s=np.array([100,500,1000])

f2,axes=plt.subplots(3,sharex=True,gridspec_kw={'height_ratios':[4,4,2]})
[ax.grid(True) for ax in axes]
ax=axes[0]
ax2=axes[1]
a3=axes[2]


ax.set_xscale('log')
ax.xaxis.set_inverted(True)
ax2.set_ylabel('Error at source [V]')
a3.set_ylabel('Points in source')
# a3.set_xlabel('Closest node to origin [m]')

a3.set_xlabel('Closest node/source radius')
a3.set_yscale('log')

ax.set_ylabel('Total error')

for xx in x0s:
    for rElec in elRs:
        for k in [0.1, 0.2, 0.3]:
            x0max=xx*rElec
            condStr="r%dx%dk%d"%(int(rElec*1e6),xx,int(10*k))
            
            if animate:
                plotter=xCell.Visualizers.ErrorGraph(plt.figure(), 
                                                     study0)
                plotter.axes[0].xaxis.set_inverted(True)
            
            def boundaryFun(coord):
                r=np.linalg.norm(coord)
                val=rElec/(r*np.pi*4)
                return val
            
            
            etots=[]
            esrc=[]
            nInSrc=[]
            rclosest=[]
            rrel=[]
            
            logmin=np.floor(np.log10(rElec/2))
            logmax=logmin+2
            
            for l0min in np.logspace(logmax,logmin):
                maxdepth=np.floor(np.log2(x0max/l0min))+2
                xmax=l0min*2**maxdepth
                study,setup=Common.makeSynthStudy(fname,
                                                  xmax=xmax,
                                                  rElec=rElec)
            
                metric=xCell.makeExplicitLinearMetric(maxdepth, 
                                                      k)
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
                
                if animate:
                    plotter.addSimulationData(setup)
                emetric,evec,_,sortr=setup.calculateErrors()
                etots.append(emetric)
                esrc.append(evec[sortr][0])
                nInSrc.append(sum(setup.nodeRoleTable==2))
                
                r=np.linalg.norm(setup.mesh.nodeCoords,axis=1)
                
                # rclose=min(r[setup.nodeRoleTable!=2])
                rclose=min(r[r!=0])
            
                rclosest.append(rclose)
                rrel.append(rclose/rElec)
                
            fbase=study.studyPath+'/'+condStr
                
            if animate:
                ani=plotter.animateStudy(fbase,fps=10)
            
            pdata={
                'condStr':condStr,
                'rclosest':rclosest,
                'rrel':rrel,
                'etots':etots,
                'esrc':esrc,
                'nInSrc':nInSrc}
            
            pickle.dump(pdata, open(fbase+'.pdata','wb'))
            # totcol='tab:orange'
            # pkcol='tab:red'
            totcol='k'
            pkcol='k'
            
            # rplot=rclosest
            rplot=rrel
    
            
            ax.plot(rplot,etots,label=condStr)
    
            ax2.plot(rplot,esrc)
    
            a3.plot(rplot,nInSrc)

ax.legend()

study.savePlot(f2, 'multiplot', '.png')
study.savePlot(f2, 'multiplot', '.eps')