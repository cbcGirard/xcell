#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBS Electrode
===============

Programmatically generate micro-macro electrode array
"""
# %%
import xcell
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv

from matplotlib.colors import to_rgba_array

xdom = 5e-2

dbody = 1.3e-3
dmicro = 17e-6
wmacro = 2e-3

pitchMacro = 5e-3

nmacro = 6  # 4-6
nmicro = 24  # 10-24

microRows = 5
microCols = 3

tpulse = 1e-3  # per phase
ipulse = 150e-6
vpulse = 1.

bbox = xdom * np.concatenate((-np.ones(3), np.ones(3)))

maxdepth = int(np.log2(xdom / dmicro)) + 2

sim = xcell.Simulation('test', bbox=bbox)


orientation = np.array([1., 0., 0.])

tipPt = np.zeros(3)
bodyL = 0.1

body = xcell.geometry.Cylinder(
    tipPt+bodyL*orientation/2, radius=dbody/2, length=bodyL, axis=orientation)

bodyMesh = pv.Cylinder(center=body.center,
                       direction=body.axis,
                       radius=body.radius,
                       height=body.length)

macroElectrodes = []
microElectrodes = []
elecMeshes = []

refPts = []
refSizes = []

for ii in range(nmacro):
    pt = tipPt+(ii+1)*pitchMacro*orientation

    geo = xcell.geometry.Cylinder(pt, dbody/2, wmacro, orientation)

    sim.addCurrentSource(0,
                         coords=pt,
                         geometry=geo)

    macroElectrodes.append(geo)

    refPts.append(geo.center)

    elecMeshes.append(pv.Cylinder(center=geo.center,
                                  direction=geo.axis,
                                  radius=geo.radius,
                                  height=geo.length,
                                  ))


for ii in range(microRows):
    rowpt = tipPt+(ii+.5)*pitchMacro*orientation

    for jj in range(microCols):
        rot = Rotation.from_rotvec(np.array([0, 2*jj/microCols*np.pi, 0]))
        microOrientation = rot.apply(0.5*dbody*np.array([0., 0., 1.]))

        geo = xcell.geometry.Disk(
            center=rowpt+microOrientation,
            radius=dmicro/2, axis=microOrientation,
            tol=0.5)

        sim.addCurrentSource(0, coords=geo.center,
                             geometry=geo)

        microElectrodes.append(geo)

        refPts.append(geo.center)
        elecMeshes.append(pv.Disc(center=geo.center,
                                  inner=0,
                                  outer=geo.radius,
                                  normal=geo.axis))


p = pv.Plotter()
# p.add_mesh(bodyMesh, color='white')
[p.add_mesh(m, color='gold') for m in elecMeshes]
p.show(auto_close=False)


sim.quickAdaptiveGrid(maxdepth)

inCyl = body.isInside(sim.mesh.nodeCoords).astype(int)
inElec = np.any([a.geometry.isInside(sim.mesh.nodeCoords)
                 for a in sim.currentSources], axis=0).astype(int)


colors = to_rgba_array([xcell.colors.NULL,
                        xcell.colors.FAINT,
                        xcell.colors.scopeColors[0]]
                       )
colors[1, -1] = 0.02

vals = inCyl+inElec


ax = xcell.visualizers.new3dPlot(bbox)

xcell.visualizers.showNodes3d(
    ax, sim.mesh.nodeCoords, vals, colors=colors[vals])

# ax.set_axis_off() #hides axes too
[a.pane.set_alpha(0) for a in [ax.xaxis, ax.yaxis, ax.zaxis]
 ]  # only hides background planes

k = 4

ax.set_xlim3d(-(k-1)*pitchMacro, (k+1)*pitchMacro)
ax.set_ylim3d(-k*pitchMacro, k*pitchMacro)
ax.set_zlim3d(-k*pitchMacro, k*pitchMacro)
ax.view_init(elev=30, azim=-135)

# %%
