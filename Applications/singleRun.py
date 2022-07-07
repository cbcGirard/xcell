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
maxdepth = 8
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

    setup.makeAdaptiveGrid(refPts=np.zeros((1,3)),
                           maxdepth=np.array(maxdepth, ndmin=1),
                           minl0Function=xcell.generalMetric,
                           coefs=np.ones(1))


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


# %%
# Interactive slice viewer (use arrow keys to change location)
sv=xcell.visualizers.SliceViewer(axis=None, setup)
# sv.nodeData=pt

# %%
# 2d image
bnd=setup.mesh.bbox[[0,3,2,4]]

arr,_=setup.getValuesInPlane()
cMap,cNorm=xcell.visualizers.getCmap(setup.nodeVoltages,forceBipolar=True)
xcell.visualizers.patchworkImage(plt.figure().gca(),
                                  arr, cMap, cNorm,
                                  extent=bnd)

ax=plt.figure().add_subplot()
xcell.visualizers.formatXYAxis(ax,bnd)
arr=xcell.visualizers.resamplePlane(ax, setup)

cMap,cNorm=xcell.visualizers.getCmap(arr.ravel(),forceBipolar=True)
xcell.visualizers.patchworkImage(ax,
                                  [arr], cMap, cNorm,
                                  extent=bnd)

_,_,edgePoints=setup.getElementsInPlane()
xcell.visualizers.showEdges2d(ax, edgePoints)


# %%
##### TOPOLOGY/connectivity
ax = xcell.visualizers.showMesh(setup)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ghost=(.0, .0, .0, 0.0)
ax.xaxis.set_pane_color(ghost)
ax.yaxis.set_pane_color(ghost)
ax.zaxis.set_pane_color(ghost)


xcell.visualizers.showEdges(ax,
                            setup.mesh.nodeCoords,
                            setup.edges,
                            setup.conductances)

bnodes = setup.mesh.getBoundaryNodes()
xcell.visualizers.showNodes3d(ax,
                              setup.mesh.nodeCoords[bnodes],
                              nodeVals=np.ones_like(bnodes),
                              colors='r')



# %%
img=xcell.visualizers.SliceSet(plt.figure(),study)
img.addSimulationData(setup,append=True)
img.getArtists(0)

# %%
# ERROR GRAPH
ptr = xcell.visualizers.ErrorGraph(plt.figure(), study)
ptr.prefs['universalPts'] = True
pdata = ptr.addSimulationData(setup)
ptr.getArtists(0, pdata)


# %%
# LOGLOG Error
P = xcell.visualizers.LogError(None, study)
P.addSimulationData(setup, True)
P.getArtists(0)
