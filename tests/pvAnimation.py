#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for creating PyVista animation with changing mesh topology.
"""

import xcell as xc
import pyvista as pv
import numpy as np
from tqdm import trange

xc.colors.use_light_style()

xdom = 1e-4
bbox = xdom * np.array([-1, -1, -1, 1, 1, 1])
ROI = 0.1 * bbox[np.array([0, 3, 1, 4])]

study = xc.Study("pvTest", bounding_box=bbox)
sim = study.new_simulation()
elec = xc.geometry.Disk(np.zeros(3), radius=1e-6, axis=np.array([0.0, 0.0, 1.0]))
elecMesh = xc.geometry.to_pyvista(elec)
sim.add_current_source(1.0, coords=np.zeros(3), radius=1e-6, geometry=elec)

regions = xc.io.Regions()
regions["Electrodes"].append(elecMesh)


# xmesh = pv.UnstructuredGrid()
sim.quick_adaptive_grid(1)
xmesh = xc.io.to_vtk(sim.mesh).slice("z")
# xmesh.cell_data['sigma'] = 1.


p = xc.visualizers.PVScene()
# p = pv.Plotter()
study.make_pv_movie(p, "pvMovieTest")
p.setup(regions.toPlane(), mesh=xmesh, simData="sigma", show_scalar_bar=False, show_edges=True)
p.planeview(ROI)

# p.add_mesh(elecMesh, color='gold')
# p.add_mesh(xmesh, show_edges=True, scalars='sigma')

depths = range(3, 14)
iterator = trange(len(depths))


for ii in iterator:
    d = depths[ii]
    sim.quick_adaptive_grid(d)
    msh = xc.io.to_vtk(sim.mesh).slice("z")
    xmesh.copy_from(msh)
    p.edgeMesh.copy_from(msh)

    iterator.set_postfix_str("%d elements" % msh.n_cells)
    p.write_frame()

p.close()
