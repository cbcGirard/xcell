#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 20:28:33 2021

@author: benoit
"""


import numpy as np
import numba as nb
import xCell
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


meshtype='adaptive'
studyPath='Results/studyTst/animateRefine/'#+meshtype

xmax=1
sigma=np.ones(3)


vMode=False
showGraphs=False
saveGraphs=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

# bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
bbox=np.append(xmax*np.zeros(3),xmax*np.ones(3))



study=xCell.SimStudy(studyPath,bbox)

l0Min=1e-6
rElec=1e-6

lastNumEl=0

k=0.5

def makeAdaptiveGrid(simulation,metric,maxdepth):

    simulation.mesh=xCell.Octree(bbox,maxdepth)

    # otree=xCell.Octree(bbox,maxDepth=maxdepth)
    simulation.mesh.refineByMetric(metric)

    # simulation.mesh.finalize()
    
    # sourceIndex=setup.mesh.coord2Index(np.zeros(3))
    # return sourceIndex

   
lastpts=np.zeros((0,3))
arts=[]
fig=plt.figure()
ax=plt.gca()

XX,YY=np.meshgrid([0,1],[0,1])
r=np.sqrt(XX**2+YY**2)

# cmap=mpl.cm.plasma
# cnorm=mpl.colors.Normalize(vmin=0, vmax=np.sqrt(2))

# im=plt.imshow(k*r,extent=[0.,1.,0.,1.],origin='lower',interpolation='bilinear')
# plt.colorbar(im)

for maxdepth in range(0,10):
    l0Param=2**(-maxdepth*0.2)
    
    setup=study.newSimulation()
    setup.mesh.elementType='Admittance'
    setup.meshtype=meshtype
    setup.mesh.minl0=2*xmax/(2**maxdepth)
    setup.ptPerAxis=1+2**maxdepth
    
    if vMode:
        setup.addVoltageSource(1,np.zeros(3),rElec)
        srcMag=1.
        srcType='Voltage'
    else:
        srcMag=4*np.pi*sigma[0]*rElec
        setup.addCurrentSource(srcMag,np.zeros(3),rElec)
        # setup.addCurrentSource(srcMag,index=sourceIndex)
        srcType='Current'

    if meshtype=='uniform':
        newNx=int(np.ceil(lastNumEl**(1/3)))
        sourceIndex=makeUniformGrid(setup,newNx+newNx%2)
    else:
        def metric(coord):
            r=np.linalg.norm(coord)
            val=k*r #1/r dependence
            # val=(l0Param*r**2)**(1/3) #current continuity
            # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
            # if val<rElec:
            #     val=rElec
            return val

        makeAdaptiveGrid(setup, metric,maxdepth)
    

    setup.startTiming("Make elements")
  
    coords=setup.mesh.nodeCoords
    
    def boundaryFun(coord):
        r=np.linalg.norm(coord)
        return rElec/(r*np.pi*4)
    
        
    setup.logTime()
    # numEl=len(setup.mesh.elements)
    
    # print('%d elem'%numEl)
    setup.finalizeMesh()
    edges,_=setup.mesh.getConductances()

    
    study.newLogEntry()#['Error','k','Depth'],[errEst,l0Param,maxdepth])
    study.saveData(setup)
    lastNumEl=len(setup.mesh.elements)
    
    centers=[]
    cvals=[]
    els=setup.mesh.elements
    for el in els:
        l0=el.l0
        center=el.origin+el.extents/2
        # centers.append(center)
        # cvals.append(l0)
        if l0>(k*np.linalg.norm(center)):
            centers.append(center)
            cvals.append(l0)
            
            
    # ccolor=cmap(cnorm(cvals))
        
    cpts=np.array(centers,ndmin=2)
    # cpts=np.array([el.origin+el.span/2 for el in setup.mesh.tree.getTerminalOctants()])
    # newpts=np.setdiff1d(cpts, lastpts)
    
    # xCell.showEdges(xCell.new3dPlot(bbox), setup.mesh.nodeCoords, edges)
    
    edgePoints=xCell.getPlanarEdgePoints(setup.mesh.nodeCoords, edges)
    
    art=xCell.showEdges2d(ax, edgePoints,edgeColors=(0.,0.,0.),alpha=1.)
    title=fig.text(0.5,.95,
                   'Split if l_0>%.2f r, depth %d'%(k,maxdepth),
                   horizontalalignment='center',verticalalignment='bottom')
    arts.append([art,title])
    ctrArt=ax.scatter(cpts[:,0],cpts[:,1],c='r')
    arts.append([ctrArt,art,title])
    # lastpts=newpts
        
ani=mpl.animation.ArtistAnimation(fig,arts,interval=1000, repeat_delay=2000,blit=False)
ani.save(os.path.join(studyPath,'%d'%(k*10)+'.mp4'),fps=1)