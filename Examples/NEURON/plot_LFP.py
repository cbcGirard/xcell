#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFP estimation with dynamic remeshing
=============================

Plot LFP from toy neurons

"""

from neuron import h  # , gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import ArtistAnimation
import xcell
from tqdm import trange

import Common
import time
import pickle
import os

from xcell import nrnutil as nUtil
from matplotlib.lines import Line2D

resultFolder = '/tmp'


h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)

mpl.style.use('fast')


nRing = 5
nSegs = 5

dmax = 8
dmin = 4

# vids = False
# post = False
nskip = 4

# Overrides for e.g. debugging in Spyder
#args.vids=True
# args.synth=False
# args.folder='Quals/polyCell'
#args.folder='Quals/monoCell'
#args.strat='depth'
#args.nSegs = 101
# args.folder='tst'
#args.nskip=1
#args.nRing=0

#%%

ring = Common.Ring(N=nRing, stim_delay=0, dendSegs=nSegs, r=175)
tstop = 40
tPerFrame = 5
barRatio=[9,1]

ivecs, isSphere, coords, rads = nUtil.getNeuronGeometry()


t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(tstop)

tv = np.array(t)*1e-3
I = 1e-9*np.array(ivecs).transpose()


I = I[::nskip]
tv = tv[::nskip]


analyticVmax = I/(4*np.pi*np.array(rads, ndmin=2))
vPeak = np.max(np.abs(analyticVmax))

imax = np.max(np.abs(I[::nskip]))


coord = np.array(coords)
xmax = 2*np.max(np.concatenate(
    (np.max(coord, axis=0), np.min(coord, axis=0))
))

#round up
xmax=xcell.util.oneDigit(xmax)
if xmax <= 0 or np.isnan(xmax):
    xmax = 1e-4


lastNumEl = 0
lastI = 0

study, setup = Common.makeSynthStudy(resultFolder, xmax=xmax)
setup.currentSources = []
studyPath = study.studyPath

dspan = dmax-dmin


tdata=None

img = xcell.visualizers.SingleSlice(None, study,
                                    tv, tdata=tdata)

for r, c, v in zip(rads, coords, I[0]):
    setup.addCurrentSource(v, c, r)

tmax = tv.shape[0]
errdicts = []

for ii in trange(0, tmax):

    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]
    vScale = np.abs(analyticVmax[ii])/vPeak

    setup.currentTime = tval

    changed = False


    for jj in range(len(setup.currentSources)):
        ival = ivals[jj]
        setup.currentSources[jj].value = ival

    # Depth strategy
    scale = dmin+dspan*vScale
    dint = np.rint(scale)
    maxdepth = np.floor(scale).astype(int)

    density = 0.2  # +0.2*dfrac

    metricCoef = 2**(-density*scale)


    netScale = 2**(-maxdepth*density)

    changed = setup.makeAdaptiveGrid(
        coord, maxdepth, xcell.generalMetric, coefs=metricCoef)

    if changed or ii == 0:
        setup.meshnum += 1
        setup.finalizeMesh()

        numEl = len(setup.mesh.elements)

        setup.setBoundaryNodes()

        v = setup.iterativeSolve()
        lastNumEl = numEl
        setup.iteration += 1

        study.saveData(setup)  # ,baseName=str(setup.iteration))
        print('%d source nodes' % sum(setup.nodeRoleTable == 2))
    else:
        # TODO: why is reset needed?
        setup.nodeRoleTable[setup.nodeRoleTable == 2] = 0

        v = setup.iterativeSolve()

    dt = time.monotonic()-t0

    lastI = ival

    study.newLogEntry(['Timestep', 'Meshnum'], [
                      setup.currentTime, setup.meshnum])

    setup.stepLogs = []

    errdict = xcell.misc.getErrorEstimates(setup)
    errdict['densities'] = density
    errdict['depths'] = maxdepth
    errdict['numels'] = lastNumEl
    errdict['dt'] = dt
    errdict['vMax'] = max(v)
    errdict['vMin'] = min(v)
    errdicts.append(errdict)


    img.addSimulationData(setup, append=True)

lists = xcell.misc.transposeDicts(errdicts)


xcell.colors.useDarkStyle()
nUtil.showCellGeo(img.axes[0])

ani=img.animateStudy('', fps=30)