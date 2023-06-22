#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:27:58 2023

@author: benoit
"""


import numpy as np
# import xcell
import pyvista as pv
from scipy.spatial import ConvexHull

stl = pv.read('fullbrain_oriented.stl')
result = pv.read('xcell2-11.vtk')

stl.rotate_x(90., inplace=True)
# pv.set_jupyter_backend(None)

pts = stl.points
ax = 2

bbox = np.hstack((np.min(pts, axis=0),
                  np.max(pts, axis=0)))
xmax = np.max(np.abs(bbox))*1.5


moldbb = stl.bounds+2*np.sign(stl.bounds)

box = pv.Box(bounds=moldbb, quads=False, level=10)

# adapted from https://gist.github.com/flutefreak7/bd621a9a836c8224e92305980ed829b9


def hull(mesh):
    hull = ConvexHull(np.array(mesh.points))
    faces = np.column_stack(
        (3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
    poly = pv.PolyData(hull.points, faces)
    return poly


# %%
k = [0.8, 0.6, 0.9]
dx = [0, -5, 0.]
p = pv.Plotter()
p.add_mesh(lhh.scale(k).translate(dx), color='red', opacity=0.8)
p.add_mesh(lh)
p.show()
# %%


maxdist = 15.
hl = hull(stl)
dst = stl.compute_implicit_distance(hl)
ok = dst.clip_scalar(value=maxdist).clip_scalar(value=-maxdist, invert=False)

ok.extract_largest().plot()
# %%
scale = [0.8, 0.8, 0.4]
slide = [0, 0, -5.]

inside = hl.scale(scale).translate(slide)
simple = stl.boolean_union(inside, progress_bar=True)

simple.plot()
# p=pv.Plotter()
# p.add_mesh(inside,color='red')
# p.add_mesh(stl,opacity=0.5,color='blue')
# p.show()

# %% try molding
moldbb = 2*np.sign(result.bounds)+np.array(result.bounds)
lbound = moldbb.copy()
rbound = moldbb.copy()
lbound[3] = 0
rbound[2] = 0

tribrain = result.extract_surface().triangulate()
lmold = pv.Box(lbound, quads=False).boolean_difference(tribrain)

# %%

vols = result.compute_cell_sizes(volume=True)
vol = vols['Volume']
depths = (np.log2(vol/min(vol))/3).astype(int)

result.cell_data['depth'] = depths

submeshes = []
for d in np.unique(depths):
    sel = depths == d

    sub = result.copy()
    sub.cell_data['sel'] = sel
    submeshes.append(sub.threshold(scalars='sel', value=0.5))


# %%
