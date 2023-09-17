#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:41:54 2022

@author: benoit
"""

import Geometry as geo
import Visualizers as viz
import numpy as np
import matplotlib as mpl


xx = np.linspace(-1, 1, 51)
X, Y, Z = np.meshgrid(xx, xx, xx)
pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()


def showBound(obj):
    cmap = mpl.colors.ListedColormap([viz.NULL, viz.BASE])

    isin = obj.is_inside(pts)

    ax = viz.new3dPlot()
    viz.show_3d_nodes(ax, pts, isin.astype(int), colormap=cmap, color_norm=mpl.colors.Normalize())


origin = np.zeros(3)
axis = -np.random.rand(3)
length = 2.0
radius = 0.5


cyl = geo.Cylinder(origin, radius, length, axis)

disk = geo.Disk(origin, radius, axis, 0.1)

# showBound(disk)
# showBound(cyl)


# Show cylinder
ptCat = cyl.is_inside(pts).astype(int)
ptCat += disk.is_inside(pts)


cmap = mpl.colors.ListedColormap([viz.NULL, viz.BASE + "10", "C0"])

ax = viz.new3dPlot()

viz.show_3d_nodes(ax, pts, ptCat, colormap=cmap, color_norm=mpl.colors.Normalize())

ax.grid(False)
[ax.spines[k].set_visible(False) for k in ax.spines]
mpl.pyplot.gcf().axes[-1].remove()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
