#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:58:31 2022

@author: benoit
"""

from neuron import h  # , gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import ArtistAnimation
# from xcell import visualizers
import xcell

import Common
import time
import pickle


from xcell import nrnutil as nUtil
from matplotlib.lines import Line2D

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)

mpl.style.use('fast')


#set nring=0 for single compartment
nRing = 5
nSegs = 5

# strat = 'depth'
strat = 'fixedMin'
# strat = 'fixedMax'

dmax = 8
dmin = 4

resultDir = 'Quals/NEURON/ring'+str(nRing)+strat

vids = False
nskip = 1



if nRing == 0:
    ring = Common.Ring(N=1, stim_delay=0, dendSegs=1, r=0)
    tstop=12
else:
    ring = Common.Ring(N=nRing, stim_delay=0, dendSegs=nSegs, r=175)
    tstop = 40

ivecs, isSphere, coords, rads = nUtil.getNeuronGeometry()
if nRing == 0:
    ivecs.pop()
    coords.pop()
    rads.pop()


t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
h.continuerun(tstop)


I = 1e-9*np.array(ivecs).transpose()
analyticVmax=I/(4*np.pi*np.array(rads,ndmin=2))
vPeak=np.max(np.abs(analyticVmax))

print('Vrange\n%.2g\t%.2g\n'%(np.min(analyticVmax), np.max(analyticVmax)))
imax = np.max(np.abs(I))


coord = np.array(coords)
xmax = 1.5*np.max(np.concatenate(
    (np.max(coord, axis=0), np.min(coord, axis=0))
))
if xmax<=0:
    xmax=1e-4


lastNumEl = 0
lastI = 0


tv = np.array(t)*1e-3


study, setup = Common.makeSynthStudy(resultDir, xmax=xmax)
setup.currentSources = []
studyPath = study.studyPath

dspan=dmax-dmin


if vids:
    img = xcell.visualizers.SingleSlice(None, study,
                                        tv)

    err=xcell.visualizers.SingleSlice(None, study,
                                      tv)
    # err.dataSrc='absErr'
    err.dataSrc='vAna'

    animators=[img,err]

    [nUtil.showCellGeo(an.axes[0]) for an in animators]
    # animators=[xcell.visualizers.ErrorGraph(None, study)]
    # animators.append(xcell.visualizers.LogError(None,study))

iEffective=[]
for r, c, v in zip(rads, coords, I[0]):
    setup.addCurrentSource(v, c, r)

tmax = tv.shape[0]
errdicts = []

for ii in range(0, tmax, nskip):

    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]
    vScale=np.abs(analyticVmax[ii])/vPeak

    setup.currentTime = tval

    changed = False
    # metrics=[]


    #TODO: fix scaling problem

    iscale = np.abs(ivals)/imax
    for jj in range(len(setup.currentSources)):
        ival = ivals[jj]
        setup.currentSources[jj].value = ival

    if strat == 'k':
        # k-param strategy
        density = 0.4*iscale
        # print('density:%.2g'%density)

    elif strat == 'depth' or strat == 'd2':
        # Depth strategy
        scale = dmin+dspan*vScale
        # dint, dfrac = divmod(np.min(scale), 1)
        dint=np.rint(scale)
        maxdepth = np.rint(scale).astype(int)
        # print('depth:%d'%maxdepth)

        # density=0.25
        if strat == 'd2':
            density = 0.5
        else:
            density = 0.2  # +0.2*dfrac

        metricCoef=2**(-density*scale)
        # metricCoef=2**(-density*dmax*iscale)

    elif strat[:5] == 'fixed':
        # Static mesh
        density = 0.2
        if strat == 'fixedMax':
            maxdepth=dmax*np.ones_like(iscale)
        elif strat == 'fixedMin':
            maxdepth=dmin*np.ones_like(iscale)
        else:
            maxdepth = 5
        metricCoef=2**(-maxdepth*density)

    netScale=2**(-maxdepth*density)


    changed = setup.makeAdaptiveGrid(coord, maxdepth, xcell.generalMetric, coefs=metricCoef)

    if changed or ii==0:
        setup.meshnum += 1
        setup.finalizeMesh()

        numEl = len(setup.mesh.elements)

        setup.setBoundaryNodes()

        v = setup.iterativeSolve()
        lastNumEl = numEl
        setup.iteration += 1

        study.saveData(setup)  # ,baseName=str(setup.iteration))
        print('%d source nodes'%sum(setup.nodeRoleTable==2))
    else:
        # vdof = setup.getDoFs()
        # v=setup.iterativeSolve(vGuess=vdof)
        # TODO: vguess slows things down?

        setup.nodeRoleTable[setup.nodeRoleTable==2]=0

        v = setup.iterativeSolve()

    dt = time.monotonic()-t0

    lastI = ival

    study.newLogEntry(['Timestep', 'Meshnum'], [
                      setup.currentTime, setup.meshnum])

    setup.stepLogs = []

    print('%d percent done' % (int(100*ii/tmax)))

    errdict = xcell.misc.getErrorEstimates(setup)
    errdict['densities'] = density
    errdict['depths'] = maxdepth
    errdict['numels'] = lastNumEl
    errdict['dt'] = dt
    errdict['vMax'] = max(np.abs(v))
    errdicts.append(errdict)

    iEffective.append(setup.nodeISources)

    if vids:
        [an.addSimulationData(setup,append=True) for an in animators]
        # err.addSimulationData(setup,append=True)
        # img.addSimulationData(setup, append=True)

lists = xcell.misc.transposeDicts(errdicts)
pickle.dump(lists, open(studyPath+'/'+strat+'.p', 'wb'))



if vids:
    # ani = img.animateStudy('volt-'+strat, fps=30.)
    # erAni=err.animateStudy('error-'+strat,fps=30.)
    erAn=[an.animateStudy(fps=30.) for an in animators]