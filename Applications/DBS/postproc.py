#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:20:51 2023

@author: benoit
"""

import xcell
import pickle
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from makeStim import getSignals
import argparse
import cmasher
from electrode import makeDBSElectrode
from xcell.geometry import toPV
import matplotlib as mpl
xcell.colors.useLightStyle()

cli = argparse.ArgumentParser()
cli.add_argument('-D', '--maxDepth', type=int,
                 help='Max depth', default=15)
cli.add_argument('-r', '--remote', action='store_true')

args = cli.parse_args()
preview = not args.remote
Depth = args.maxDepth
Depth = 16

plotRelative = True
normalize = True

fstemFix = 'full-d%dx1' % Depth
fstemAdapt = 'adaptive-'+fstemFix

moviename = 'comparison'
if plotRelative:
    moviename+'-relative'

meshname = 'sim0'

body, microE, macroE = makeDBSElectrode()
rsave = pv.read('../../Examples/Geometry/composite.vtm')
regions = xcell.io.Regions()
regions['Conductors'] = rsave['Conductors']
regions['Insulators'].append(toPV(body))
regions['Electrodes'].extend([toPV(E) for E in macroE])

# for later clipping to XY coords of hippocampus
ROI = regions['Conductors'].bounds[:4]

# set sim boundaries to 2x hippocampus bounding box
bbox = 2*np.max(np.abs(ROI))*np.concatenate((-np.ones(3), np.ones(3)))

channels = getSignals(24, 50e-3)
tvec = np.unique(np.concatenate([c.times for c in channels]))

voltrange = xcell.visualizers.ScaleRange()

datas = []
for fs in [fstemFix, fstemAdapt]:
    study = xcell.SimStudy(os.path.join(os.getcwd(), fs), bbox)
    info = study.load(fs+'info')
    volts = study.load(fs+'Volts')
    data = []
    recon = tqdm.trange(len(info), desc='Reconstructing')
    for ii in recon:
        meshname = info[ii]['meshName']
        v = volts[ii]
        voltrange.update(v)

        msh = pv.read(os.path.join(fs, meshname+'.vtk'))
        if ii == 0:
            msh.point_data['voltage'] = np.zeros(msh.n_points)
            vmesh = msh.copy()
        else:
            msh.point_data['voltage'] = v.copy()
        # print('%d\t%d' % (len(v), msh.n_points))

        slic = msh.slice(normal='z')

        info[ii]['mesh'] = slic
        info[ii]['voltage'] = slic['voltage']
        # data.append(slic['voltage'])
    datas.append(info)


# calc difference
dataFix, dataAdapt = datas

fixDict = xcell.misc.transposeDicts(dataFix)
adaptDict = xcell.misc.transposeDicts(dataAdapt)

# f, ax = plt.subplots()
# ax.plot(fixDict['Total time [Wall]'], label='Static')
# ax.plot(adaptDict['Total time [Wall]'], label='Dynamic')
# ax.set_xlabel('Timestep')
# ax.set_ylabel('Computational time [s]')
# ax.legend()
# study.savePlot(f, 'timesteps')


diffs = []

absdifrange = xcell.visualizers.ScaleRange()
relDiffRange = xcell.visualizers.ScaleRange()
amprange = xcell.visualizers.ScaleRange()
vrange = xcell.visualizers.ScaleRange()
intDiffs = []
dels = []
# for dF, dA in zip(dataFix, dataAdapt):
for ii in range(len(dataFix)):
    fmesh = dataFix[ii]['mesh']
    amesh = dataAdapt[ii]['mesh']
    difmesh = fmesh.sample(amesh)
    vrange.update(fmesh['voltage'])
    vrange.update(amesh['voltage'])

    absDiff = difmesh['voltage']-fmesh['voltage']

    # correct for change between insulator/electrode
    absDiff[np.logical_or(difmesh['voltage'] == 0.,
                          fmesh['voltage'] == 0.)] = 0

    absdifrange.update(absDiff)
    dels.extend(absDiff.tolist())

    difmesh.point_data['Absolute Difference'] = absDiff.copy()

    ampdiff = np.abs(absDiff)
    amprange.update(ampdiff)
    ampval = np.abs(difmesh['voltage'])
    difmesh.point_data['ampdiff'] = ampdiff
    difmesh.point_data['ampval'] = ampval
    intmesh = difmesh.integrate_data()
    interr = intmesh['ampdiff']/intmesh['ampval']
    if np.isfinite(interr):
        intDiffs.append(interr[0])
    else:
        intDiffs.append(0)

    relErr = np.abs(100 * absDiff/fmesh['voltage'])
    relErr[np.logical_or(np.isinf(relErr), np.isnan(relErr))] = 0.
    # relDiffRange[0] = min(relDiffRange.min(), relDiffRange[0])
    # relDiffRange[1] = max(relDiffRange.max(), relDiffRange[1])
    relDiffRange.update(relErr)

    difmesh.point_data['Relative Difference'] = relErr.copy()
    diffs.append(difmesh)

# f, ax = plt.subplots()
# ax.plot(intDiffs)
# ax.set_xlabel('Time step')
# ax.set_ylabel('Normalized difference')

# %% Plots

with mpl.rc_context({
    'lines.markersize': 2.5,
    'lines.linewidth': 1,
    'figure.figsize': [3.25, 4],
    'font.size': 10,
    'legend.fontsize': 9,
    'axes.grid': False,
    'axes.prop_cycle': plt.cycler(color=plt.colormaps['tab10'].colors)
}):
    f, axes = plt.subplots(3,
                           sharex=True,
                           gridspec_kw={'height_ratios': [5, 5, 2]}
                           )
    axes[1].plot(tvec, intDiffs)
    axes[1].set_ylabel('Normalized error')

    axes[0].plot(tvec, fixDict['Total time [Wall]'], label='Static')
    axes[0].plot(tvec, adaptDict['Total time [Wall]'][:-1], label='Dynamic')
    axes[0].set_ylabel('Wall time [s]')
    axes[0].legend()

    for ch, ii in zip(channels, np.arange(len(channels))):
        axes[2].scatter(np.array(ch.times), ii*np.ones_like(ch.times),
                        marker='|')
    axes[2].set_ylabel('Channel')
    axes[2].set_xlabel('Timestep')
    axes[2].set_yticks([])
    axes[2].set_xlim(0., 2.)

    xcell.visualizers.engineerTicks(axes[2], xunit='s')

    f.align_ylabels()
    # [a.set_xticks([]) for a in axes[:-1]]

    # study.savePlot(f, 'timeSteps')

    f2, ax2 = plt.subplots(figsize=[3.25, 2.5])
    ax2.bar([0, 1], [sum(fixDict['Total time [Wall]']),
                     sum(adaptDict['Total time [Wall]'])],
            tick_label=['Static', 'Dynamic'])
    ax2.set_ylabel('Wall time [s]')
    # study.savePlot(f2, 'Time Comparison')


stdErr = np.std(dels)

# %%

p = xcell.visualizers.PVScene(time=tvec)

# p.open_movie(fstem+'.mp4')
study.makePVmovie(p, filename=moviename)

VBAR = {
    'vertical': True,
    'height': 0.75,
    'position_x': 0.05,
    'position_y': 0.2,
    'title_font_size': 20,
    'label_font_size': 20,

}
# cma = mcmap['bwr']
# cma = xcell.colors.CM_BIPOLAR
# cma=mcmap['seismic']

mesh = diffs[0].copy()

pargs = {}
if plotRelative:
    # diffrange = np.array([0., 6*stdErr])
    # diffrange = np.array([0., max(intDiffs)])
    mag = np.floor(np.log10(vrange.max))

    if normalize:
        # diffrange = (1e-6, 1.)
        ncol = 3
        diffrange = (10**(1-ncol), 1.)
        # VBAR['n_labels'] = 3
        # colormap = 'YlOrRd'
        # colormap = cmasher.chroma_r
        colormap = 'Reds'

    else:
        diffrange = 10**mag*np.ceil(vrange.max/10**mag)*np.array([1e-3, 1.])

    scalar = 'ampdiff'
    VBAR['fmt'] = '%.0e'
    VBAR['title'] = 'Amplitude\nDifference'
    pargs['log_scale'] = True
else:
    # diffrange = voltrange.get(forceSymmetric=True)
    diffrange = 2*stdErr*np.array([-1, 1])
    scalar = 'Absolute Difference'
    colormap = 'seismic'

p.setup(regions.toPlane(), mesh=mesh,
        show_edges=True, scalar_bar_args=VBAR,
        clim=diffrange,
        lighting=False,
        simData=scalar,
        cmap=colormap,
        opacity='linear',
        line_width=2,
        **pargs
        )  # , symlog=symlog)

# p.add_mesh(mesh, scalars='voltage', clim=vrange,
#            show_edges=True, cmap=cma,
#            scalar_bar_args=VBAR, symlog=symlog)

# orient for 2d image
p.planeview(ROI)


# mesh['voltage'] = data[121]

# if not preview:
# p.show(auto_close=False)

# %%

viz = tqdm.trange(len(tvec), desc='Animating')
for ii in viz:
    t = tvec[ii]
    if ii+1 == len(tvec):
        tnext = 2.
    else:
        tnext = tvec[ii+1]

    mesh.copy_from(diffs[ii])
    if normalize:
        mesh[scalar] /= vrange.max

    while t < tnext:
        p.setTime(t)
        p.write_frame()
        study.savePVimage(p,
                          os.path.join(moviename, moviename+'frame%03d' % ii))
        t += 0.005
        break


p.close()
