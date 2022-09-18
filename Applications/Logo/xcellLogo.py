#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:22:35 2022

Generates the xcell logo from point cloud


@author: benoit
"""

# import xcell
import pandas
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

csv = 'pts.csv'
ptsFile = 'logo.xpts'

bbox = np.array([0, 0, -1, 4, 4, 1])

color = 'red'


# %%

f = open('xcelldense.ngc', 'r')
txt = f.read()
f.close()

regex = re.compile('X.*Y[\d\.]*')
txtpts = regex.findall(txt)
ctext = open(csv, 'w')
extract = ',0.0\n'.join(txtpts)
ctext.write(extract.replace('Y', '').replace('X', '').replace(' ', ','))
ctext.close()


pts = pandas.read_csv(csv).to_numpy()
pickle.dump(pts, open(ptsFile, 'wb'))

# %%
# Generate meshes from points
pts = pickle.read(open(ptsFile, 'rb'))

# Optionally visualize guide points
# x,y,_=np.hsplit(pts,3)
# plt.scatter(x,y)

fig, ax = plt.subplots()
xcell.visualizers.formatXYAxis(ax, bbox[np.array([0, 3, 1, 4])])
ax.axis('Off')

setup = xcell.Simulation('', bbox)

meshPts = []
artists = []
for d in range(0, 13):

    metrics = xcell.makeScaledMetrics(pts, 1, d, density=0.0)

    setup.makeAdaptiveGrid(metrics, d)
    setup.finalizeMesh()

    # xcell.visualizers.showMesh(setup)

    _, _, elPts = setup.getElementsInPlane()
    meshPts.append(elPts)


pickle.dump(meshPts, open('logoMesh.p', 'wb'))

# %%
# Make logo image and animation

fig, ax = plt.subplots()
xcell.visualizers.formatXYAxis(ax, bbox[np.array([0, 3, 1, 4])])
ax.axis('Off')

if color == 'red':
    col = xcell.visualizers.ACCENT_DARK
    artists = [[xcell.visualizers.showEdges2d(ax, p, edgeColors=col, alpha=0.2, linewidth=1.5),
                xcell.visualizers.showEdges2d(ax, p, edgeColors='r', alpha=0.05, linewidth=0.5)] for p in meshPts[:13]]
else:
    col = xcell.visualizers.BASE
    # col=xcell.visualizers.LINE
    artists = [[xcell.visualizers.showEdges2d(
        ax, p, edgeColors=col, alpha=0.2, linewidth=1.5)] for p in meshPts[:13]]


ani = xcell.visualizers.ArtistAnimation(fig, artists, interval=125)

outFile = 'logo'

ani.save(outFile+'.mp4', fps=8, dpi=300)
fig.savefig(outFile+'.png', dpi=300)
