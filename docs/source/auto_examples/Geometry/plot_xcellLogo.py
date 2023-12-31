#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logo
=============
Generates the xcell logo from point cloud

Original text layout in Inkscape, text bounds resampled using Roughen path effect with all displacements=0, then exported with Path to Gcode extension.

Original canvas bounds are ((0,32), (0,32))

"""

import xcell
import pandas
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import time

csv = 'pts.csv'
ptsFile = 'logo.xpts'

convertGcode=False
showOriginalPoints=False

bbox = np.array([0, 0, -32, 32, 32, 32])

xcell.colors.useDarkStyle()
color = 'base'
logoFPS = 6

# %%
# Convert gcode to array of points
# ---------------------------------------
# 

if convertGcode:

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
    okPts = ~np.any(np.isnan(pts),axis=1)
    pickle.dump(pts[okPts], open(ptsFile, 'wb'))

# %%
# Generate meshes from points
# ---------------------------------
#

pts = pickle.load(open(ptsFile, 'rb'))

tstart=time.monotonic()
setup = xcell.Simulation('', bbox)


meshPts = []
artists = []

npts=pts.shape[0]
for d in range(0, 12):

    depths=d*np.ones(npts,dtype=int)
    co=np.ones(npts)

    setup.makeAdaptiveGrid(refPts=pts,
                           maxdepth=depths,
                           coefs=co,
                           minl0Function=xcell.generalMetric,
                           coarsen=False)
    setup.finalizeMesh()

    _, _, elPts = setup.getElementsInPlane()
    meshPts.append(elPts)

t_tot=time.monotonic()-tstart

pickle.dump(meshPts, open('logoMesh.p', 'wb'))

# %%
# Make logo image and animation
# -------------------------------
# 

dpi=100

with plt.rc_context({'figure.figsize':[19.2, 10.8],
                     'figure.dpi':dpi,
                     'toolbar':'None',
                     }):

    fig, ax = plt.subplots()
    ax.set_xlim(1.,31.)
    ax.set_ylim(1., 23.)
    ax.axis('Off')
    ax.margins(0)

    col = xcell.colors.BASE
    artists = [[xcell.visualizers.showEdges2d(
        ax, p, edgeColors=col, alpha=0.2, linewidth=1.5)] for p in meshPts[:13]]

    #pad ending frame
    artists.append(artists[-1])
    artists.append(artists[-1])

    ani = xcell.visualizers.ArtistAnimation(fig, artists, interval=1000//logoFPS)

    outFile = 'logo'

    ani.save(outFile+'.mp4', fps=logoFPS, dpi=dpi)
    fig.savefig(outFile+'.png', dpi=dpi)

    tlogo=len(artists)/logoFPS
    print('%.2f second logo made in %.0f seconds'%(tlogo, t_tot))

    if showOriginalPoints:
        # Optionally visualize guide points
        x,y,_=np.hsplit(pts,3)
        plt.scatter(x,y)
