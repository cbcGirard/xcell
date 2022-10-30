#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:48:40 2022

For-fun demonstration of multiresolution meshing from an existing surface mesh.

@author: benoit
"""

# %%

import numpy as np
import xcell
import pyvista as pv

viz = False

stl = pv.read('fullbrain_oriented.stl')

pv.set_jupyter_backend(None)

pts = stl.points
ax=2

bbox = np.hstack((np.min(pts, axis=0),
                  np.max(pts, axis=0)))



xmax = np.max(np.abs(bbox))*1.5


mindepth = 2
maxdepth = 9


#%%
setup = xcell.Simulation('brain', bbox, True)

#interpolate by bounds
idealDepth = mindepth+(maxdepth-mindepth)*(pts[:, 2]-bbox[2])/(bbox[5]-bbox[2])
metfun=xcell.generalMetric


setup.makeAdaptiveGrid(refPts=pts, maxdepth=idealDepth.astype(
    int), minl0Function=metfun, coefs=0.2*np.ones_like(idealDepth),
    coarsen=False)

setup.finalizeMesh()


#%%
xmesh=xcell.io.toVTK(setup.mesh)

xmesh.save('xmesh%d%d.vtk'%(mindepth, maxdepth))
inside=xmesh.select_enclosed_points(stl,tolerance=1e-9)
result=inside.threshold(0.5).extract_largest()

fstem='xcell%d-%d'%(mindepth, maxdepth)
result.save(fstem+'.vtk')

obj=result.extract_surface()
pv.save_meshio(fstem+'.obj', obj)



#%%

if viz:
    p=pv.Plotter()
    p.add_mesh(result, color='red', opacity=0.5)
    p.add_mesh(stl, color = 'blue', opacity = 0.75)
    p.show()
