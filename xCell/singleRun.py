#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:02:59 2022
Regularization tests
@author: benoit
"""


import numpy as np
import numba as nb
import xCell
import matplotlib.pyplot as plt


meshtype='adaptive'
# studyPath='Results/studyTst/miniCur/'#+meshtype
studyPath='Results/studyTst/dual/'

elementType='Admittance'

xmax=1e-4
maxdepth=11

sigma=np.ones(3)


vMode=False
showGraphs=False
generate=False
saveGraphs=False

dual=True
regularize=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))
# if dual:
#     bbox+=xmax*2**(-maxdepth)


study=xCell.SimStudy(studyPath,bbox)

l0Min=1e-6
rElec=1e-6

lastNumEl=0
meshTypes=["adaptive","uniform"]


l0Param=2**(-maxdepth*0.2)
# l0Param=0.2

setup=study.newSimulation()
setup.mesh.elementType=elementType
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
    srcType='Current'

if meshtype=='uniform':
    newNx=int(np.ceil(lastNumEl**(1/3)))
    nX=newNx+newNx%2
    setup.makeUniformGrid(newNx+newNx%2)
    print('uniform, %d per axis'%nX)
else:
    def metric(coord,l0Param=l0Param):
        r=np.linalg.norm(coord)
        val=l0Param*r #1/r dependence
        # val=(l0Param*r**2)**(1/3) #current continuity
        # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
        # if val<rElec:
        #     val=rElec
        
        if (r+val)<rElec:
            val=rElec/2
        return val
    
    # def metric(coord):
    #     r=np.linalg.norm(coord)
    #     # val=l0Param*r #1/r dependence
    #     val=(1e-7*r**2)**(1/3) #current continuity
    #     # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
    #     # if val<rElec:
    #     #     val=rElec
        
    #     if (r+val)<rElec:
    #         val=rElec/2
    #     return val
    
    setup.makeAdaptiveGrid(metric,maxdepth)


  

def boundaryFun(coord):
    r=np.linalg.norm(coord)
    return rElec/(r*np.pi*4)

    

# setup.finalizeMesh()
# mcoords=setup.mesh.nodeCoords
# medges=setup.edges


setup.mesh.elementType='Face'
setup.asDual=True
setup.finalizeMesh()

setup.setBoundaryNodes(boundaryFun)

# v=setup.solve()
v=setup.iterativeSolve(None,1e-9)

setup.getMemUsage(True)
setup.printTotalTime()

setup.startTiming('Estimate error')
errEst,_,_,_=setup.calculateErrors()#srcMag,srcType,showPlots=showGraphs)
print('error: %g'%errEst)
setup.logTime()


setup.applyTransforms()

pt,val=setup.getUniversalPoints()
coords=xCell.util.indexToCoords(pt, study.bbox[:3],study.span, maxdepth)

setup.mesh.nodeCoords=coords
setup.nodeVoltages=val

ax=plt.gca()
sv=xCell.Visualizers.SliceViewer(ax, setup)
sv.nodeData=pt



ax=xCell.Visualizers.new3dPlot(study.bbox)
cmap,cnorm=xCell.Visualizers.getCmap(val,forceBipolar=True)
xCell.Visualizers.showNodes3d(ax, coords, val,cMap=cmap,cNorm=cnorm)



# bnd=setup.mesh.bbox[[0,3,2,4]]
# arr,_=setup.getValuesInPlane()
# cMap,cNorm=xCell.Visualizers.getCmap(setup.nodeVoltages)
# xCell.Visualizers.patchworkImage(plt.figure().gca(), 
#                                  arr, cMap, cNorm, 
#                                  extent=bnd)

ptr=xCell.Visualizers.ErrorGraph(plt.figure(),study)

# ptr=xCell.Visualizers.SliceSet(plt.figure(),study)
pdata=ptr.addSimulationData(setup)
ptr.getArtists(0,pdata)

# setup.mesh.elementType='Admittance'
# setup.finalizeMesh()

# els=setup.getElementsInPlane()
# eg,_=setup.mesh.getConductances(els)
# medges=xCell.util.renumberIndices(eg,setup.mesh.indexMap)
# mcoords=setup.mesh.nodeCoords

# xCell.Visualizers.showMesh(setup)
# ax=plt.gca()
# # ax=xCell.Visualizers.new3dPlot(study.bbox)

# # xCell.Visualizers.showNodes3d(ax, coords, imap,colors=plt.cm.get_cmap('tab10')(imap))
# xCell.Visualizers.showEdges(ax, mcoords, medges)

# bnodes=setup.mesh.getBoundaryNodes()
# xCell.Visualizers.showNodes3d(ax,
#                               setup.mesh.nodeCoords[bnodes],
#                               nodeVals=np.ones_like(bnodes),
#                               colors='r')


# # # # xCell.Visualizers.showMesh(setup)
# # eg=xCell.Visualizers.ErrorGraph(plt.figure(), study)
# # eg.addSimulationData(setup)
# # eg.getArtists()

# img=xCell.Visualizers.SliceSet(plt.figure(),study)
# img.addSimulationData(setup)
# img.getArtists()