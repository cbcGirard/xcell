#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single timestep
=====================

Illustrates setting up a simulation and solving at a single time step

"""

import numpy as np
import xcell
import matplotlib.pyplot as plt


# %%
# Simulation preferences

# Misc parameters
xcell.colors.useLightStyle()
studyPath = '/dev/null'

# options = uniform, adaptive
meshtype = 'adaptive'

maxdepth = 10  # Maximum successive splits allowed for octree mesh
nX = 10  # Number of elements along an axis for a uniform mesh

# options: Admittance, Face, FEM
elementType = 'Admittance'
dual = True
regularize = False

# options: analytical, ground
boundaryType = 'ground'

fixedVoltageSource = False  # otherwise, simulate current injection

# %%
# Setup simulation


xmax = 1e-4  # domain boundary
rElec = 1e-6  # center source radius

sigma = np.ones(3)

bbox = np.append(-xmax*np.ones(3), xmax*np.ones(3))
study = xcell.SimStudy(studyPath, bbox)

setup = study.newSimulation()
setup.mesh.elementType = elementType
setup.meshtype = meshtype

if fixedVoltageSource:
    setup.addVoltageSource(xcell.signals.Signal(1), np.zeros(3), rElec)
    srcMag = 1.
    srcType = 'Voltage'
else:
    srcMag = 4*np.pi*sigma[0]*rElec
    setup.addCurrentSource(xcell.signals.Signal(srcMag), np.zeros(3), rElec)
    srcType = 'Current'

if meshtype == 'uniform':
    setup.makeUniformGrid(nX)
    print('uniform, %d per axis' % nX)
else:
    setup.makeAdaptiveGrid(refPts=np.zeros((1, 3)),
                           maxdepth=np.array(maxdepth, ndmin=1),
                           minl0Function=xcell.generalMetric,
                           # coefs=np.array(2**(-0.2*maxdepth), ndmin=1))
                           coefs=np.array(0.2, ndmin=1))

if boundaryType == 'analytical':
    boundaryFun = None
else:
    def boundaryFun(coord):
        r = np.linalg.norm(coord)
        return rElec/(r*np.pi*4)


setup.finalizeMesh()

setup.setBoundaryNodes(boundaryFun, sigma=1)

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
# SliceViewer
# ----------------------
# Interactive slice viewer (use arrow keys to change location within ipython session)
#

sv = xcell.visualizers.SliceViewer(axis=None, sim=setup)

# %%
# 2d image
bnd = setup.mesh.bbox[[0, 3, 2, 4]]

arr, _ = setup.getValuesInPlane()
cMap, cNorm = xcell.visualizers.getCmap(setup.nodeVoltages, forceBipolar=True)
xcell.visualizers.patchworkImage(plt.figure().gca(),
                                 arr, cMap, cNorm,
                                 extent=bnd)

# %%
#

ax = plt.figure().add_subplot()
xcell.visualizers.formatXYAxis(ax, bnd)
arr = xcell.visualizers.resamplePlane(ax, setup)

cMap, cNorm = xcell.visualizers.getCmap(arr.ravel(), forceBipolar=True)
xcell.visualizers.patchworkImage(ax,
                                 [arr], cMap, cNorm,
                                 extent=bnd)

_, _, edgePoints = setup.getElementsInPlane()
xcell.visualizers.showEdges2d(ax, edgePoints)


# %%
# TOPOLOGY/connectivity
ax = xcell.visualizers.showMesh(setup)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ghost = (.0, .0, .0, 0.0)
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
# SliceSet
# --------------------
#

# sphinx_gallery_thumbnail_number = 5
img = xcell.visualizers.SliceSet(plt.figure(), study)
img.addSimulationData(setup, append=True)
img.getArtists(0)

# %%
# ErrorGraph
# ------------------
#

ptr = xcell.visualizers.ErrorGraph(plt.figure(), study)
ptr.prefs['universalPts'] = True
pdata = ptr.addSimulationData(setup)
ptr.getArtists(0, pdata)


# %%
# LogError
# -----------
#

P = xcell.visualizers.LogError(None, study)
P.addSimulationData(setup, True)
P.getArtists(0)
