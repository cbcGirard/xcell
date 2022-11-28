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


stl = pv.read('fullbrain_oriented.stl')

# pv.set_jupyter_backend(None)

# pts = stl.points
# ax=2

# bbox = np.hstack((np.min(pts, axis=0),
#                   np.max(pts, axis=0)))
# xmax = np.max(np.abs(bbox))*1.5

# mindepth = 2
# maxdepth = 9

# setup = xcell.Simulation('brain', bbox, True)

# #interpolate by bounds
# idealDepth = mindepth+(maxdepth-mindepth)*(pts[:, 2]-bbox[2])/(bbox[5]-bbox[2])
# metfun=xcell.generalMetric

# setup.makeAdaptiveGrid(refPts=pts, maxdepth=idealDepth.astype(
#     int), minl0Function=metfun, coefs=0.2*np.ones_like(idealDepth),
#     coarsen=False)

# setup.finalizeMesh()


# # %%
# # Segment mesh and visualize
# # -------------------------------
# # 
# xmesh=xcell.io.toVTK(setup.mesh)

# xmesh.save('xmesh%d%d.vtk'%(mindepth, maxdepth))
# inside=xmesh.select_enclosed_points(stl,tolerance=1e-9)
# result=inside.threshold(0.5).extract_largest()

# fstem='xcell%d-%d'%(mindepth, maxdepth)
# result.save(fstem+'.vtk')

# obj=result.extract_surface()
# pv.save_meshio(fstem+'.obj', obj)



result=pv.read('xcell2-9.vtk')

thm=pv.themes.DarkTheme()
thm.background=pv.Color(xcell.colors.DARK,opacity=0)
offwhite=pv.Color(xcell.colors.OFFWHITE, opacity=1.)


pv.set_plot_theme(thm)

p=pv.Plotter(off_screen=True)
p.add_mesh(stl, color = 'blue', opacity = 0.75)

viewup=[0.2,-1.,0.]

p.add_mesh(result, show_edges=False, color=offwhite)

p.enable_eye_dome_lighting()

p.show(auto_close=False)

# # path = p.generate_orbital_path(factor=1.5, n_points=144, viewup=viewup, shift=-0.2)
# path = p.generate_orbital_path(factor=1.5, n_points=32, viewup=viewup, shift=-0.2)

# p.open_movie("orbit.mp4")

# p.orbit_on_path(path, write_frames=True, viewup=viewup, step=0.05, progress_bar=True)
