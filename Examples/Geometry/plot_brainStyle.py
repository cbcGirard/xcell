#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stylized brain
========================
Multiresolution meshing from an existing surface mesh.

Original mesh, generated from MRI data, is available at `<https://3dprint.nih.gov/discover/3DPX-000320>`_

"""

import numpy as np
import xcell
import pyvista as pv
import tqdm
from vtk import VTK_QUAD

stl = pv.read('full_oriented_simpler.stl')

# pv.set_jupyter_backend(None)

pts = stl.points
ax = 2

bbox = np.hstack((np.min(pts, axis=0),
                  np.max(pts, axis=0)))
xmax = np.max(np.abs(bbox))*1.5

mindepth = 2
max_depth = 6

setup = xcell.Simulation('brain', xmax*np.sign(bbox), True)

# interpolate by bounds
idealDepth = mindepth+(max_depth-mindepth)*(pts[:, 2]-bbox[2])/(bbox[5]-bbox[2])
metfun = xcell.general_metric

setup.make_adaptive_grid(ref_pts=pts, max_depth=idealDepth.astype(
    int), min_l0_function=metfun, coefs=0.2*np.ones_like(idealDepth),
    coarsen=False)

setup.finalize_mesh()

# %%
adj = setup.mesh.get_element_adjacencies()
# reg = xcell.io.Regions()
# stl.cell_data['sigma'] = 1.
# reg['Conductors'].append(stl)

# reg.assign_sigma(setup.mesh, default_sigma=0)

# %% alt sigma calc

xmesh = xcell.io.to_vtk(setup.mesh)


xmesh.save('xmesh%d%d.vtk' % (mindepth, max_depth))
inside = xmesh.cell_centers().select_enclosed_points(stl, tolerance=1e-9)


for el, s in zip(setup.mesh.elements, inside['SelectedPoints']):
    el.sigma = s
# %%
okfaces = []

quadorders = np.array([[0, 4, 6, 2],
                       [1, 3, 7, 5],
                       [0, 1, 5, 4],
                       [2, 6, 7, 3],
                       [0, 2, 3, 1],
                       [4, 5, 7, 6]])

for ii in tqdm.trange(len(setup.mesh.elements), desc='Checking faces'):
    el = setup.mesh.elements[ii]
    if el.sigma > 0:
        neighbors = adj[ii]

        for jj in range(6):
            neighbor = neighbors[jj]

            for nei in neighbor:
                if nei.sigma == 0:
                    if nei.depth > el.depth:

                        otherFaceInd = jj+(-1)**jj
                        inds = np.flip(nei.vertices[quadorders[otherFaceInd]])
                    else:
                        inds = el.vertices[quadorders[jj]]

                    globalind = [setup.mesh.inverse_index_map[idx] for idx in inds]
                    okfaces.append(globalind)


fc = np.array(okfaces)

usedNodes = np.unique(fc.ravel())
revNodes = {}
for ii, n in enumerate(usedNodes):
    revNodes[n] = ii

newpts = setup.mesh.node_coords[usedNodes]

newFaces = np.array([revNodes[n]
                    for n in fc.ravel()]).reshape((fc.shape[0], 4))

cells = np.hstack(
    (4*np.ones((newFaces.shape[0], 1), dtype=np.uint64), newFaces.astype(np.uint64))).ravel()
cellTypes = [VTK_QUAD]*newFaces.shape[0]

qmesh = pv.UnstructuredGrid(cells, cellTypes, newpts)
qmesh.save('quad.vtk')

qg = qmesh.compute_cell_sizes(length=False, volume=False)
qg.set_active_scalars('Area')
# qg.plot(style='wireframe')

# %%
# Segment mesh and visualize
# -------------------------------
#

fstem = 'xcell%d-%d' % (mindepth, max_depth)

result = inside.threshold(
    value=0.5, scalars='SelectedPoints').extract_largest()
result.save(fstem+'.vtk')

# obj = result.extract_surface()
# pv.save_meshio(fstem+'.obj', obj)
pv.save_meshio('quads.obj', qmesh)


# %%
# Generate animation
# ------------------------------
#

# to generate from premeshed file
# result = pv.read('xcell2-9.vtk')

thm = pv.themes.DarkTheme()
thm.background = pv.Color(xcell.colors.DARK, opacity=0)
offwhite = pv.Color(xcell.colors.OFFWHITE, opacity=1.)


pv.set_plot_theme(thm)

p = pv.Plotter(off_screen=True)
p.add_mesh(stl, color='blue', opacity=0.75)

viewup = [0.2, -1., 0.]

p.add_mesh(result, show_edges=False, color=offwhite)

p.enable_eye_dome_lighting()

p.show(auto_close=False)

path = p.generate_orbital_path(
    factor=1.5, n_points=32, viewup=viewup, shift=-0.2)

p.open_movie("orbit.mp4")

p.orbit_on_path(path, write_frames=True, viewup=viewup,
                step=0.1, progress_bar=True)
