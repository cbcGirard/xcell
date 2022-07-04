#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:02:59 2022
Regularization tests
@author: benoit
"""


import numpy as np
# import numba as nb
import xcell
import matplotlib.pyplot as plt


# meshtype = 'uniform'
meshtype= 'adaptive'
# studyPath='Results/studyTst/miniCur/'#+meshtype
studyPath = '/dev/null'

elementType = 'Admittance'

# elementType='Face'

xmax = 1e-4
maxdepth = 18
nX = 10

sigma = np.ones(3)


vMode = False
showGraphs = False
generate = False
saveGraphs = False

dual = True
regularize = False

vsrc = 1.
isrc = vsrc*4*np.pi*sigma*1e-6

bbox = np.append(-xmax*np.ones(3), xmax*np.ones(3))
# bbox=np.append(np.zeros(3),xmax*np.ones(3))
# if dual:
#     bbox+=xmax*2**(-maxdepth)


study = xcell.SimStudy(studyPath, bbox)

l0Min = 1e-6
rElec = 1e-6

lastNumEl = 0
meshTypes = ["adaptive", "uniform"]


l0Param = 2**(-maxdepth*0.2)
# l0Param=0.2

setup = study.newSimulation()
setup.mesh.elementType = elementType
setup.meshtype = meshtype
# setup.mesh.minl0=2*xmax/(2**maxdepth)
# setup.ptPerAxis=1+2**maxdepth

if vMode:
    setup.addVoltageSource(1, np.zeros(3), rElec)
    srcMag = 1.
    srcType = 'Voltage'
else:
    srcMag = 4*np.pi*sigma[0]*rElec
    setup.addCurrentSource(srcMag, np.zeros(3), rElec)
    srcType = 'Current'

if meshtype == 'uniform':
    setup.makeUniformGrid(nX)
    print('uniform, %d per axis' % nX)
else:

    # metric=xcell.makeBoundedLinearMetric(l0min=2e-6,
    #                                      l0max=1e-5,
    #                                      domainX=xmax)

    metric = [xcell.makeExplicitLinearMetric(maxdepth, 0.2)]

    setup.makeAdaptiveGrid(metric, maxdepth)


boundaryFun = None
# def boundaryFun(coord):
#     r=np.linalg.norm(coord)
#     return rElec/(r*np.pi*4)


setup.finalizeMesh()

setup.setBoundaryNodes(boundaryFun, sigma=1)

# v=setup.solve()
v = setup.iterativeSolve(None, 1e-9)
setup.applyTransforms()
setup.getMemUsage(True)
setup.printTotalTime()


setup.startTiming('Estimate error')
# srcMag,srcType,showPlots=showGraphs)
errEst, arErr, _, _, _ = setup.calculateErrors()
print('error: %g' % errEst)
setup.logTime()


# Tweaking error metrics
# setup.nodeVoltages=np.zeros_like(setup.nodeVoltages)
# r=np.linalg.norm(setup.mesh.nodeCoords,axis=1)
# va,_=setup.analyticalEstimate(r)
# setup.nodeVoltages=.2*va[0]


# pt,val=setup.getUniversalPoints()
# coords=xcell.util.indexToCoords(pt, study.bbox[:3],study.span)

# setup.mesh.nodeCoords=coords
# setup.nodeVoltages=val

# ax=plt.gca()
# sv=xcell.visualizers.SliceViewer(ax, setup)
# sv.nodeData=pt


# ax=xcell.visualizers.new3dPlot(study.bbox)
# cmap,cnorm=xcell.visualizers.getCmap(val,forceBipolar=True)
# xcell.visualizers.showNodes3d(ax, coords, val,cMap=cmap,cNorm=cnorm)


# # 2d image
# bnd=setup.mesh.bbox[[0,3,2,4]]

# # arr,_=setup.getValuesInPlane()
# # cMap,cNorm=xcell.visualizers.getCmap(setup.nodeVoltages,forceBipolar=True)
# # xcell.visualizers.patchworkImage(plt.figure().gca(),
# #                                   arr, cMap, cNorm,
# #                                   extent=bnd)

# ax=plt.figure().add_subplot()
# xcell.visualizers.formatXYAxis(ax,bnd)
# arr=xcell.visualizers.resamplePlane(ax, setup)

# cMap,cNorm=xcell.visualizers.getCmap(arr.ravel(),forceBipolar=True)
# xcell.visualizers.patchworkImage(ax,
#                                   [arr], cMap, cNorm,
#                                   extent=bnd)

# _,_,edgePoints=setup.getElementsInPlane()
# xcell.visualizers.showEdges2d(ax, edgePoints)


# # setup.mesh.elementType='Admittance'
# setup.finalizeMesh()

# els=setup.getElementsInPlane()
# eg,_=setup.mesh.getConductances(els)
# medges=xcell.util.renumberIndices(eg,setup.mesh.indexMap)
# mcoords=setup.mesh.nodeCoords


# ##### TOPOLOGY/connectivity
# ax = xcell.visualizers.showMesh(setup)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ghost=(.0, .0, .0, 0.0)
# ax.xaxis.set_pane_color(ghost)
# ax.yaxis.set_pane_color(ghost)
# ax.zaxis.set_pane_color(ghost)


# xcell.visualizers.showEdges(ax,
#                             setup.mesh.nodeCoords,
#                             setup.edges,
#                             setup.conductances)

# bnodes = setup.mesh.getBoundaryNodes()
# xcell.visualizers.showNodes3d(ax,
#                               setup.mesh.nodeCoords[bnodes],
#                               nodeVals=np.ones_like(bnodes),
#                               colors='r')


# # # # xcell.visualizers.showMesh(setup)
# # eg=xcell.visualizers.ErrorGraph(plt.figure(), study)
# # eg.addSimulationData(setup)
# # eg.getArtists()

# img=xcell.visualizers.SliceSet(plt.figure(),study)
# img.addSimulationData(setup)
# img.getArtists()


# # ERROR GRAPH
# ptr = xcell.visualizers.ErrorGraph(plt.figure(), study)
# ptr.prefs['universalPts'] = True
# pdata = ptr.addSimulationData(setup)
# ptr.getArtists(0, pdata)


# _,basicAna,basicErr,_=setup.estimateVolumeError(basic=True)
# _,advAna,advErr,_=setup.estimateVolumeError(basic=False)

# errBasic=sum(basicErr)/sum(basicAna)
# errAdv=sum(advErr)/sum(advAna)


# es,err,ana,sr,r=setup.calculateErrors()


# print('Error metrics:\nbasic vol:%g\nadv vol:%g\narea%g'%(errBasic,errAdv,es))

# # LOGLOG Error
# P = xcell.visualizers.LogError(None, study)
# P.addSimulationData(setup, True)
# P.getArtists(0)
