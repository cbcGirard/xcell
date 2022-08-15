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
import os

from xcell import nrnutil as nUtil
from matplotlib.lines import Line2D

import argparse

h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)

mpl.style.use('fast')


# #set nring=0 for single compartment
# nRing = 5
# nSegs = 5

# synth=False

# strat = 'depth'
# #strat = 'fixedMin'
# # strat = 'fixedMax'

dmax = 8
dmin = 4

# vids = False
# post = False
# nskip = 2

cli = argparse.ArgumentParser()
cli.add_argument('-f', '--folder',
                 help='path from main results folder', default='polyCell/')
cli.add_argument('-n', '--nRing', type=int,
                 help='number of cells (default %(default)s) 0->single compartment', default=5)
cli.add_argument('-s', '--nSegs', type=int,
                 help='segments per dendrite (default %(default)s', default=5)
cli.add_argument('-k', '--nskip', type=int,
                 help='timesteps to skip', default=2)
cli.add_argument('--strat', help='meshing approach', choices=[
                 'fixedMin', 'depth', 'fixedMax', 'depthExt'], default='fixedMin')
cli.add_argument('-v', '--vids', help='generate animation',
                 action='store_true')
cli.add_argument('-p', '--post', help='generate summary plots',
                 action='store_true')
cli.add_argument(
    '-S', '--synth', help='use dummy pulses instead of biopyhsical sources', action='store_true')


args = cli.parse_args()

# Overrides for e.g. debugging in Spyder
#args.vids=True
#args.synth=True
#args.folder='tmp'

#%%
if args.strat == 'depthExt':
    dmax += 4

if args.nRing == 0:
    ring = Common.Ring(N=1, stim_delay=0, dendSegs=1, r=0)
    tstop = 12
else:
    ring = Common.Ring(N=args.nRing, stim_delay=0, dendSegs=args.nSegs, r=175)
    tstop = 40

ivecs, isSphere, coords, rads = nUtil.getNeuronGeometry()
if args.nRing == 0:
    ivecs.pop()
    coords.pop()
    rads.pop()

if args.synth:
    ncomp = len(rads)
    tv = np.linspace(0, 4+2*ncomp)

    def synthPulse(pulseT):
        posPhase = np.exp(-10*(tv-1.5*pulseT-2)**2)
        negPhase = np.exp(-10*(tv-1.5*pulseT-3)**2)

        return 1e-9*(posPhase-negPhase)

    I = np.array([synthPulse(k) for k in range(ncomp)]).transpose()
else:
    t = h.Vector().record(h._ref_t)
    h.finitialize(-65 * mV)
    h.continuerun(tstop)

    tv = np.array(t)*1e-3
    I = 1e-9*np.array(ivecs).transpose()


I = I[::args.nskip]
tv = tv[::args.nskip]

analyticVmax = I/(4*np.pi*np.array(rads, ndmin=2))
vPeak = np.max(np.abs(analyticVmax))

print('Vrange\n%.2g\t%.2g\n' % (np.min(analyticVmax), np.max(analyticVmax)))
imax = np.max(np.abs(I[::args.nskip]))


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


resultFolder = os.path.join(args.folder, args.strat)

study, setup = Common.makeSynthStudy(resultFolder, xmax=xmax)
setup.currentSources = []
studyPath = study.studyPath

dspan = dmax-dmin


if args.vids:
    img = xcell.visualizers.SingleSlice(None, study,
                                        tv)

    err = xcell.visualizers.SingleSlice(None, study,
                                        tv)
    err.dataSrc = 'absErr'
    # err.dataSrc='vAna'

    animators = [img, err]
    aniNames = ['volt-', 'error-']

    # animators = [img]
    # aniNames = ['volt-']


    # animators=[xcell.visualizers.ErrorGraph(None, study)]
    # animators.append(xcell.visualizers.LogError(None,study))

iEffective = []
for r, c, v in zip(rads, coords, I[0]):
    setup.addCurrentSource(v, c, r)

tmax = tv.shape[0]
errdicts = []


# %%
# run simulatios
for ii in range(0, tmax):

    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]
    vScale = np.abs(analyticVmax[ii])/vPeak

    setup.currentTime = tval

    changed = False
    # metrics=[]


    for jj in range(len(setup.currentSources)):
        ival = ivals[jj]
        setup.currentSources[jj].value = ival

    if args.strat == 'k':
        # k-param strategy
        density = 0.4*vScale
        # print('density:%.2g'%density)

    elif args.strat[:5] == 'depth' or args.strat == 'd2':
        # Depth strategy
        scale = dmin+dspan*vScale
        # dint, dfrac = divmod(np.min(scale), 1)
        dint = np.rint(scale)
        maxdepth = np.floor(scale).astype(int)
        # print('depth:%d'%maxdepth)

        # density=0.25
        if args.strat == 'd2':
            density = 0.5
        else:
            density = 0.2  # +0.2*dfrac

        metricCoef = 2**(-density*scale)
        # metricCoef=2**(-density*dmax*vScale)

    elif args.strat[:5] == 'fixed':
        # Static mesh
        density = 0.2
        if args.strat == 'fixedMax':
            maxdepth = dmax*np.ones_like(vScale)
            dmin=dmax
        elif args.strat == 'fixedMin':
            maxdepth = dmin*np.ones_like(vScale)
            dmax=dmin
        else:
            maxdepth = 5
        metricCoef = 2**(-maxdepth*density)

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
        # vdof = setup.getDoFs()
        # v=setup.iterativeSolve(vGuess=vdof)
        # TODO: vguess slows things down?

        setup.nodeRoleTable[setup.nodeRoleTable == 2] = 0

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
    errdict['vMax'] = max(v)
    errdict['vMin'] = min(v)
    errdicts.append(errdict)

    iEffective.append(setup.nodeISources)

    if args.vids:
        [an.addSimulationData(setup, append=True) for an in animators]
        # err.addSimulationData(setup,append=True)
        # img.addSimulationData(setup, append=True)

lists = xcell.misc.transposeDicts(errdicts)
lists['depths']='Depth [%d:%d]'%(dmin,dmax)
pickle.dump(lists, open(studyPath+'/'+args.strat+'.pcr', 'wb'))


if args.vids:

    # for lite in ['','-lite']:
    #     if lite=='':
    #         xcell.colors.useLightStyle()
    #     else:
    #         xcell.colors.useDarkStyle()
    #         anis=[an.animateStudy(l+args.strat,fps=30) for an,l in zip(animators, aniNames)]

    for an,l in zip(animators, aniNames):
        xcell.colors.useDarkStyle()
        nUtil.showCellGeo(an.axes[0])

        fname=l+args.strat
        an.animateStudy(fname, fps=30)

        xcell.colors.useLightStyle()
        alite=an.copy()

        nUtil.showCellGeo(alite.axes[0])

        #get closest frames to 5ms intervals
        frameNs=[int(f*len(alite.dataSets)/tstop) for f in np.arange(0,tstop, 5)]
        artists=[alite.getArtists(ii) for ii in frameNs]
        alite.animateStudy(fname+'-lite', fps=30, artists=artists,  vectorFrames=np.arange(len(frameNs)))


# %%
if args.post:
    folderstem = os.path.dirname(studyPath)

    def getdata(strat):
        fname = os.path.join(folderstem, strat, strat+'.pcr')
        data = pickle.load(open(fname, 'rb'))
        return data

    labels=[l for l in os.listdir(folderstem) if os.path.isdir(os.path.join(folderstem,l))]
    data=[getdata(l) for l in labels]


    for fmt in ['.png','.svg','.eps']:
        for lite in ['','-lite']:
            if lite=='':
                xcell.colors.useLightStyle()
            else:
                xcell.colors.useDarkStyle()

            f, ax = plt.subplots(3, gridspec_kw={'height_ratios': [5, 5, 2]})
            ax[2].plot(tv, I)

            for d, l in zip(data, labels):
                ax[0].semilogy(tv, np.abs(d['volErr']), label=d['depths'])
                # ax[0].plot(tv,  np.abs(d['intErr']), label=l)
                ax[1].plot(tv[1:], d['dt'][1:])

            ax[0].legend()
            ax[0].set_ylabel('Error')
            ax[1].set_ylabel('Wall time')
            ax[2].set_ylabel('Activity')
            ax[2].set_xlabel('Simulated time')

            xcell.visualizers.engineerTicks(ax[2], xunit='s')
            [a.set_xticks([]) for a in ax[:-1]]



            f2,ax=plt.subplots(1,1)
            aright=ax.twinx()

            # [a.set_yscale('log') for a in [ax,aright]]

            ttime=[sum(l['dt']) for l in data]
            terr=[sum(np.abs(l['intErr'])) for l in data]
            # terr=[sum(np.abs(l['intErr']))/sum(np.abs(l['intAna'])) for l in data]
            # terr=[sum(np.abs(l['SSE']))/sum(np.abs(l['SSTot'])) for l in data]

            categories=[d['depths'] for d in data]
            barpos=np.arange(len(categories))
            barw=0.25

            tart=ax.bar(barpos-barw/2,height=ttime, width=barw, color='C0', label='Time')
            ax.set_ylabel('Total simulation time')

            eart=aright.bar(barpos+barw/2, height=terr, width= barw, color='C1', label='Error')
            aright.set_ylabel('Total simulation error')

            ax.set_xticks(barpos)
            ax.set_xticklabels(categories)

            ax.legend(handles=[tart, eart])


            study.savePlot(f2, os.path.join(
                folderstem, 'ring%dsummary%s' % (args.nRing, lite)), ext=fmt)

            study.savePlot(f, os.path.join(folderstem, 'ring%dcomparison%s' % (args.nRing, lite)), ext=fmt)
