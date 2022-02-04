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
import util
import Visualizers


studyPath='/home/benoit/smb4k/ResearchData/Results/studyTst/algoDemo/'#+meshtype
# fname='splitIllustration'
fname='faceIllustration'
dual=True



showSrcCircuit=True
lastGen=5
fullCircle=True


xmax=1
sigma=np.ones(3)


vMode=False
showGraphs=False
saveGraphs=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

rElec=xmax/10
l0Min=1e-6
rElec=1e-1

lastNumEl=0

k=0.5

if fullCircle:
    bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
else:
    bbox=np.append(xmax*np.zeros(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)






arts=[]
fig=plt.figure()
cm=Visualizers.CurrentPlot(fig, study,fullarrow=True,showInset=False)
ax=cm.fig.axes[0]


if fullCircle:
    tht=np.linspace(0,2*np.pi)
else:
    tht=np.linspace(0,np.pi/2)
arcX=rElec*np.cos(tht)
arcY=rElec*np.sin(tht)

# ax.plot(arcX,arcY,'r--',label='Source boundary')
src=ax.fill(arcX,arcY,color=mpl.cm.plasma(1.0),alpha=0.5,label='Source')
ax.legend(handles=src)

XX,YY=np.meshgrid([0,1],[0,1])
r=np.sqrt(XX**2+YY**2)

for maxdepth in range(1,lastGen+1):
    l0Param=2**(-maxdepth*0.2)
    
    setup=study.newSimulation()
    if dual:
        setup.mesh.elementType='Face'
    else:
        setup.mesh.elementType='Admittance'
 
    def metric(coord):
        r=np.linalg.norm(coord)
        val=k*r #1/r dependence
        # val=(l0Param*r**2)**(1/3) #current continuity
        # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
        # if val<rElec:
        #     val=rElec
        return val

    setup.makeAdaptiveGrid(metric,maxdepth)
        

    setup.finalizeMesh(regularize=False, dualMesh=False)
    edges,_=setup.mesh.getConductances()
    coords=setup.mesh.nodeCoords
    
    # coords=setup.getCoords()
    # edges=setup.getEdges()
    # coords,edges=setup.getMeshGeometry()




    edgePoints=Visualizers.getPlanarEdgePoints(coords, edges)
    
    art=Visualizers.showEdges2d(ax, edgePoints,edgeColors=(0.,0.,0.),alpha=1.)
    title=Visualizers.animatedTitle(fig, 'Split if l_0>%.2f r, depth %d'%(k,maxdepth))
    arts.append([art,title])
    
    if maxdepth!=lastGen:
        #show dot at center of elements needing split
        
        centers=[]
        cvals=[]
        els=setup.mesh.elements
        for el in els:
            l0=el.l0
            center=el.origin+el.span/2
            # centers.append(center)
            # cvals.append(l0)
            if l0>(k*np.linalg.norm(center)):
                centers.append(center)
                cvals.append(l0)
            
        cpts=np.array(centers,ndmin=2)
        
        ctrArt=ax.scatter(cpts[:,0],cpts[:,1],c='r')
        arts.append([ctrArt,art,title])
    
    
if showSrcCircuit:
    #find nodes inside source, and map their location to center
    # inSrc=np.linalg.norm(setup.mesh.nodeCoords,axis=1)<=rElec
    
    viewer=Visualizers.SliceViewer(ax,setup)
    
    setup.addCurrentSource(1, np.zeros(3),rElec)
    setup.insertSourcesInMesh()
    setup.finalizeMesh(regularize=False, dualMesh=dual)
    
    setup.setBoundaryNodes()
    setup.iterativeSolve()
    
    inSrc=setup.nodeRoleTable==2
    # inPlane=setup.mesh.nodeCoords[:,-1]==0
    # inSrc=np.logical_and(inPlane, inSrc)
    
    # oldEdges=setup.mesh.edges
    edgeInSrc=np.sum(inSrc[setup.edges], axis=1)
    # srcEdges=oldEdges[edgeInSrc]
    
    # edgeColors=np.zeros((oldEdges.shape[0],3))
    # edgeColors[edgeInSrc,0]=1
    
    # pts=Visualizers.getPlanarEdgePoints(setup.mesh.nodeCoords, oldEdges)
    # edgeArt=Visualizers.showEdges2d(ax, pts, oldEdges, colors=edgeColors)
    
    # sX,sY=np.hsplit(setup.mesh.nodeCoords[inSrc,:-1], 2)
    # nodeArt=plt.scatter(sX,sY,marker='.',c='r')
    

    nodeColors=np.array([
        [0,0,0,0],
        [1,0,0,1]],dtype=float)
    edgeColors=np.array([
        [0, 0, 0, 0.05],
        [1,0.5, 0, 0.25],
        [1, 0, 0, 1]])

    viewer.edgeData=edgeInSrc
    viewer.nodeData=inSrc.astype(int)
    
    eCol=edgeColors[edgeInSrc.astype(int)]
    
    if not dual:
        edgeArt=viewer.showEdges(colors=eCol)
        nodeArt=viewer.showNodes(inSrc,colors=nodeColors[inSrc.astype(int)])
        
        title=Visualizers.animatedTitle(fig, 'Combine nodes inside source')
        arts.append([nodeArt,edgeArt, title])
        arts.append([nodeArt,edgeArt, title])
    
    
    # elCoords=setup.mesh.nodeCoords.copy()
    # elCoords[inSrc]=np.zeros(3)
    elCoords=setup.getCoords(orderType='electrical')
    elEdges=setup.getEdges(orderType='electrical')

    # srcRemap=np.arange(setup.mesh.nodeCoords.shape[0])
    # srcRemap[inSrc]=0
    # setup.mesh.nodeCoords[0]=np.zeros(3)
    
    # edgeRemap=srcRemap[setup.edges]
    touchesSrc=np.any(inSrc[setup.edges],axis=1).astype(int)
    equColor=edgeColors[touchesSrc]
    # 
    # srcEdge=edgeRemap[touchesSrc]
    # restEdge=edgeRemap[~touchesSrc]
    
    # rePts=Visualizers.getPlanarEdgePoints(elCoords[:,:-1], setup.edges)    
    # reArt=Visualizers.showEdges2d(ax, rePts, colors=equColor)
    
    viewer.topoType='electrical'
    viewer.setPlane()
    reArt=viewer.showEdges(colors=equColor)
    title=Visualizers.animatedTitle(fig, 'Equivalent circuit')
    arts.append([reArt,title])
    arts.append([reArt,title])


    cm.addSimulationData(setup)
    endArts=cm.getArtists()
    endArts[0].append(Visualizers.animatedTitle(fig, 'Current distribution'))
    arts.extend(endArts)
    
ani=cm.animateStudy(fname,artists=arts)
# ani=mpl.animation.ArtistAnimation(fig,arts,interval=1000, repeat_delay=2000,blit=False)
# ani.save(os.path.join(studyPath,'%d'%(k*10)+'.mp4'),fps=1)