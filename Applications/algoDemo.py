#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 20:28:33 2021

@author: benoit
"""

# %%
import re
from matplotlib.markers import MarkerStyle
import numpy as np
import numba as nb
import xcell
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from xcell import util
from xcell import visualizers
import Common

lite = True
asDual = False
studyPath = '/home/benoit/smb4k/ResearchData/Results/Quals/algoDemo/'
fname = 'overview'
if asDual:
    fname += '-dual'

if lite:
    xcell.colors.useLightStyle()
    mpl.rcParams.update({'figure.figsize':[2.0, 2.0],
                         'font.size':10.,
                         'lines.markersize':5.,
                         })
    fname += '-lite'

showSrcCircuit = True
lastGen = 4
fullCircle = True

xmax = 1

rElec = xmax/5


k = 0.5

if fullCircle:
    bbox = np.append(-xmax*np.ones(3), xmax*np.ones(3))
else:
    bbox = np.append(xmax*np.zeros(3), xmax*np.ones(3))

study, setup = Common.makeSynthStudy(studyPath, rElec=rElec, xmax=xmax)


arts = []
fig = plt.figure(constrained_layout=True)


cm = visualizers.CurrentPlot(
    fig, study, fullarrow=True, showInset=False, showAll=asDual)
cm.prefs['colorbar'] = False
cm.prefs['title'] = False
cm.prefs['logScale'] = True
ax = cm.fig.axes[0]

ax.set_aspect('equal')
#hide axes
ax.set_xticks([])
ax.set_yticks([])
plt.title("spacer",color=xcell.colors.NULL)



if fullCircle:
    tht = np.linspace(0, 2*np.pi)
else:
    tht = np.linspace(0, np.pi/2)
arcX = rElec*np.cos(tht)
arcY = rElec*np.sin(tht)

src = ax.fill(arcX, arcY, color=mpl.cm.plasma(1.0), alpha=0.5, label='Source')
# ax.legend(handles=src)

noteColor = xcell.colors.ACCENT_DARK


for maxdepth in range(1, lastGen+1):
    l0Param = 2**(-maxdepth*0.2)


    setup.makeAdaptiveGrid(refPts=np.zeros((1,3)),
                           maxdepth=np.array(maxdepth, ndmin=1),
                           minl0Function=xcell.generalMetric,
                           # coefs=np.array(2**(-0.2*maxdepth), ndmin=1))
                           coefs=np.array(k, ndmin=1))

    setup.finalizeMesh(regularize=False)
    # edges,_,_=setup.mesh.getConductances()
    coords = setup.mesh.nodeCoords

    # coords=setup.getCoords()
    # edges=setup.getEdges()
    coords, edges = setup.getMeshGeometry()
    # edges=[setup.mesh.inverseIdxMap[n] for n in ]

    edgePoints = visualizers.getPlanarEdgePoints(coords, edges)

    # ,edgeColors=visualizers.FAINT,alpha=1.)
    art = visualizers.showEdges2d(ax, edgePoints)
    title = visualizers.animatedTitle(fig,
    # title = ax.set_title(
        r'Split if $\ell_0$>%.2f r, depth %d' % (k, maxdepth))
    arts.append([art, title])

    if maxdepth != lastGen:
        # show dot at center of elements needing split

        centers = []
        cvals = []
        els = setup.mesh.elements
        for el in els:
            l0 = el.l0
            center = el.origin+el.span/2

            if l0 > (k*np.linalg.norm(center)):
                centers.append(center)
                cvals.append(l0)

        cpts = np.array(centers, ndmin=2)

        ctrArt = ax.scatter(cpts[:, 0], cpts[:, 1],
                            c=noteColor, marker='o')
        arts.append([ctrArt, art, title])


# %%
if showSrcCircuit:
    # outside, inside source
    nodeColors = np.array([
        [0, 0, 0, 0],
        [0.6, 0, 0, 1]], dtype=float)

    # edges outside, crossing, and fully inside source
    edgeColors = np.array([
        xcell.colors.FAINT,
        [1, 0.5, 0, 1.],
        [1, 0, 0, 1]])

    if asDual:
        nodeColors[0, -1] = 0.1
        # edgeColors[0,-1]=0.05
    else:
        edgeColors[0, -1] /= 2

    finalMesh = art
    if asDual:
        finalMesh.set_alpha(0.25)
        setup.mesh.elementType = 'Face'
    setup.finalizeMesh()

    # hack to get plane elements only
    els, pts, _ = setup.getElementsInPlane()

    m2 = xcell.meshes.Mesh(bbox)
    m2.elements = els

    setup.mesh = m2
    if asDual:
        # visualizers.showEdges2d(ax, edgePoints,alpha=0.5)
        setup.mesh.elementType = 'Face'
    setup.finalizeMesh()

    setup.setBoundaryNodes()
    setup.iterativeSolve()

    inSrc = setup.nodeRoleTable == 2

    # oldEdges=setup.edges
    edgePtInSrc = np.sum(inSrc[setup.edges], axis=1)
    # srcEdges=oldEdges[edgePtInSrc==2]

    mergeColors = edgeColors[edgePtInSrc]

    # mergePts=visualizers.getPlanarEdgePoints(setup.mesh.nodeCoords, setup.edges)

    sX, sY = np.hsplit(setup.mesh.nodeCoords[inSrc, :-1], 2)
    nodeArt = plt.scatter(sX, sY,marker='*', c=noteColor)

    title = visualizers.animatedTitle(fig,
    # title = ax.set_title(
                                      'Combine nodes inside source')
    artset = [nodeArt, title]  # ,finalMesh]

    if asDual:
        artset.append(finalMesh)
    else:
        mergePts = setup.mesh.nodeCoords[setup.edges, :-1]
        edgeArt = visualizers.showEdges2d(ax, mergePts, colors=mergeColors)
        artset.append(edgeArt)

    for ii in range(2):
        arts.append(artset)
# %%

    # replace with single source node
    srcIdx = inSrc.nonzero()[0][0]
    setup.edges[inSrc[setup.edges]] = srcIdx
    setup.mesh.nodeCoords[srcIdx] = setup.currentSources[0].coords

    nTouchingSrc = np.sum(inSrc[setup.edges], axis=1)

    equivColors = mergeColors[nTouchingSrc]

    # eqPts=visualizers.getPlanarEdgePoints(setup.mesh.nodeCoords, setup.edges)
    eqPts = setup.mesh.nodeCoords[setup.edges, :-1]
    reArt = visualizers.showEdges2d(ax, eqPts, colors=equivColors)
    ctrArt = ax.scatter(0, 0, c=noteColor, marker='*')

    # viewer.topoType='electrical'
    # viewer.setPlane(showAll=asDual)
    # reArt=viewer.showEdges(colors=equColor)
    title = visualizers.animatedTitle(fig,
    # title=ax.set_title(
                                      'Equivalent circuit')

    eqArtists = [reArt, title, ctrArt]
    if asDual:
        eqArtists.append(finalMesh)

    for ii in range(3):
        arts.append(eqArtists)

    cm.addSimulationData(setup, append=True)
    endArts = cm.getArtists(0)
    endArts.append(visualizers.animatedTitle(fig,
    # endArts.append(ax.set_title(
    'Current distribution'))

    if asDual:
        endArts.append(finalMesh)

    for ii in range(5):
        arts.append(endArts)


ani = cm.animateStudy(fname, artists=arts)
