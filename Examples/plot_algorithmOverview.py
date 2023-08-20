#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation overview
===================================

A simplified view of the meshing process, electrical network generation, and solution

"""

import numpy as np
import xcell as xc
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from xcell import util
from xcell import visualizers
import Common_nongallery

# %%
# Setup animation
# -----------------

lite = False
asDual = False
study_path = "algoDemo"
fname = "overview"
if asDual:
    fname += "-dual"

if lite:
    fname += "-lite"
    xc.colors.use_light_style()
    mpl.rcParams.update(
        {
            "figure.figsize": [2.0, 2.0],
            "font.size": 10.0,
            "lines.markersize": 5.0,
        }
    )
else:
    mpl.rcParams.update({"figure.figsize": [2.0, 2.0], "font.size": 9, "figure.dpi": 500})

showSrcCircuit = True
lastGen = 4
fullCircle = True

xmax = 1

rElec = xmax / 5


k = 0.5

if fullCircle:
    bbox = np.append(-xmax * np.ones(3), xmax * np.ones(3))
else:
    bbox = np.append(xmax * np.zeros(3), xmax * np.ones(3))

study, setup = Common_nongallery.makeSynthStudy(study_path, rElec=rElec, xmax=xmax)


arts = []
fig = plt.figure(constrained_layout=True)


cm = visualizers.CurrentPlot(fig, study, fullarrow=True, showInset=False, showAll=asDual)
cm.prefs["colorbar"] = False
cm.prefs["title"] = False
cm.prefs["logScale"] = True
ax = cm.fig.axes[0]

ax.set_aspect("equal")
# hide axes
ax.set_xticks([])
ax.set_yticks([])
plt.title("spacer", color=xc.colors.NULL)


if fullCircle:
    tht = np.linspace(0, 2 * np.pi)
else:
    tht = np.linspace(0, np.pi / 2)
arcX = rElec * np.cos(tht)
arcY = rElec * np.sin(tht)

src = ax.fill(arcX, arcY, color=mpl.cm.plasma(1.0), alpha=0.5, label="Source")

noteColor = xc.colors.ACCENT_DARK

# %%
# Subdivide mesh
# ----------------
#
# The target element size is proportional to its distance from the source. Elements larger than their target size are recursively split until all are smaller than their respective target.

for max_depth in range(1, lastGen + 1):
    l0Param = 2 ** (-max_depth * 0.2)

    setup.make_adaptive_grid(
        ref_pts=np.zeros((1, 3)),
        max_depth=np.array(max_depth, ndmin=1),
        min_l0_function=xc.general_metric,
        coefs=np.array(k, ndmin=1),
    )

    setup.finalize_mesh(regularize=False)
    coords = setup.mesh.node_coords

    coords, edges = setup.get_mesh_geometry()

    edge_points = visualizers.get_planar_edge_points(coords, edges)

    art = visualizers.show_2d_edges(ax, edge_points)
    title = visualizers.animated_title(
        fig,
        # title = ax.set_title(
        r"Split if $\ell_0$>%.2f r, depth %d" % (k, max_depth),
    )
    arts.append([art, title])

    if max_depth != lastGen:
        # show dot at center of elements needing split

        centers = []
        cvals = []
        els = setup.mesh.elements
        for el in els:
            l0 = el.l0
            center = el.origin + el.span / 2

            if l0 > (k * np.linalg.norm(center)):
                centers.append(center)
                cvals.append(l0)

        cpts = np.array(centers, ndmin=2)

        ctrArt = ax.scatter(cpts[:, 0], cpts[:, 1], c=noteColor, marker="o")
        arts.append([ctrArt, art, title])

# %%
# Electrical network
# ----------------------
#
# Each element has discrete conducances between its vertices. All nodes inside a source are combined into a single electrical node
#
if showSrcCircuit:
    # outside, inside source
    nodeColors = np.array([[0, 0, 0, 0], [0.6, 0, 0, 1]], dtype=float)

    # edges outside, crossing, and fully inside source
    edge_colors = np.array([xc.colors.FAINT, [1, 0.5, 0, 1.0], [1, 0, 0, 1]])

    if asDual:
        nodeColors[0, -1] = 0.1
    else:
        edge_colors[0, -1] /= 2

    finalMesh = art
    if asDual:
        finalMesh.set_alpha(0.25)
        setup.mesh.element_type = "Face"
    setup.finalize_mesh()

    # hack to get plane elements only
    els, pts, _ = setup.get_elements_in_plane()

    m2 = xc.meshes.Mesh(bbox)
    m2.elements = els

    setup.mesh = m2
    if asDual:
        setup.mesh.element_type = "Face"
    setup.finalize_mesh()

    setup.set_boundary_nodes()
    setup.solve()

    inSrc = setup.node_role_table == 2

    edgePtInSrc = np.sum(inSrc[setup.edges], axis=1)

    mergeColors = edge_colors[edgePtInSrc]

    sX, sY = np.hsplit(setup.mesh.node_coords[inSrc, :-1], 2)
    nodeArt = plt.scatter(sX, sY, marker="*", c=noteColor)

    title = visualizers.animated_title(fig, "Combine nodes inside source")
    artset = [nodeArt, title]

    if asDual:
        artset.append(finalMesh)
    else:
        mergePts = setup.mesh.node_coords[setup.edges, :-1]
        edgeArt = visualizers.show_2d_edges(ax, mergePts, colors=mergeColors)
        artset.append(edgeArt)

    for ii in range(2):
        arts.append(artset)

    # replace with single source node
    srcIdx = inSrc.nonzero()[0][0]
    setup.edges[inSrc[setup.edges]] = srcIdx
    source = setup.current_sources[0]
    setup.mesh.node_coords[srcIdx] = source.geometry.center

    nTouchingSrc = np.sum(inSrc[setup.edges], axis=1)

    equivColors = mergeColors[nTouchingSrc]

    eqPts = setup.mesh.node_coords[setup.edges, :-1]
    reArt = visualizers.show_2d_edges(ax, eqPts, colors=equivColors)
    ctrArt = ax.scatter(0, 0, c=noteColor, marker="*")

    title = visualizers.animated_title(fig, "Equivalent circuit")

    eqArtists = [reArt, title, ctrArt]
    if asDual:
        eqArtists.append(finalMesh)

    for ii in range(3):
        arts.append(eqArtists)

    # %%
    # Solve for node voltages
    # -------------------------
    #
    # The

    cm.add_simulation_data(setup, append=True)
    endArts = cm.get_artists(0)
    endArts.append(visualizers.animated_title(fig, "Current distribution"))

    if asDual:
        endArts.append(finalMesh)

    for ii in range(5):
        arts.append(endArts)


ani = cm.animate_study(fname, artists=arts)
