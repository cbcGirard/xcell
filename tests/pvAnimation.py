#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:00:57 2023

@author: benoit
"""

import xcell as xc
import pyvista as pv
import numpy as np
from tqdm import trange

xc.colors.useLightStyle()

xdom = 1e-4
bbox = xdom * np.array([-1, -1, -1, 1, 1, 1])
ROI = 0.1*bbox[np.array([0, 3, 1, 4])]

study = xc.SimStudy('pvTest', boundingBox=bbox)
sim = study.newSimulation()
elec = xc.geometry.Disk(np.zeros(3),
                        radius=1e-6,
                        axis=np.array([0., 0., 1.]))
elecMesh = xc.geometry.toPV(elec)
sim.addCurrentSource(1., coords=np.zeros(3),
                     radius=1e-6,
                     geometry=elec)

regions = xc.io.Regions()
regions['Electrodes'].append(elecMesh)


# xmesh = pv.UnstructuredGrid()
sim.quickAdaptiveGrid(1)
xmesh = xc.io.toVTK(sim.mesh).slice('z')
# xmesh.cell_data['sigma'] = 1.


p = xc.visualizers.PVScene()
# p = pv.Plotter()
study.makePVmovie(p, 'pvMovieTest')
p.setup(regions.toPlane(), mesh=xmesh,
        simData='sigma', show_scalar_bar=False,
        show_edges=True)
p.planeview(ROI)

# p.add_mesh(elecMesh, color='gold')
# p.add_mesh(xmesh, show_edges=True, scalars='sigma')

depths = range(3, 14)
iterator = trange(len(depths))


for ii in iterator:
    d = depths[ii]
    sim.quickAdaptiveGrid(d)
    msh = xc.io.toVTK(sim.mesh).slice('z')
    xmesh.copy_from(msh)
    p.edgeMesh.copy_from(msh)

    iterator.set_postfix_str('%d elements' % msh.n_cells)
    p.write_frame()

p.close()
