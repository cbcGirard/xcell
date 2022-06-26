#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:00:57 2022

@author: benoit
"""

import xcell as xc
import numpy as np
import Common as com
from neuron import h  # , gui
import neuron.units as nUnit

import matplotlib.pyplot as plt
import pickle


domX = 250e-6

stepViz = True
save = True

pairStim = True
analytic = False
if analytic:
    stepViz = False

# https://doi.org/10.1109/TBME.2018.2791860
# 48um diameter +24um insulation; 1ms, 50-650uA
# 2.5-6.5 Ohm-m is 0.15-0.4 S/m

# weighted mean of whole-brain conductivity [S/m], as of 5/14/22
# from https://github.com/Head-Conductivity/Human-Head-Conductivity
Sigma = 0.3841
# Sigma=1

Imag = -150e-6
tstop = 20.
tpulse = 1.
tstart = 5.
rElec = 24e-6

elecY = 50


study, _ = com.makeSynthStudy(
    'NEURON/Stim2/Dual_fixed_y'+str(elecY), xmax=domX,)


if stepViz:
    viz = xc.visualizers.SliceSet(None, study,
                                  prefs={
                                      'showError': False,
                                      'showInsets': False,
                                      'relativeError': False,
                                      'logScale': True,
                                      'showNodes': True,
                                      'fullInterp': True})
else:
    viz = None
    # zoom in
    # resetBounds(viz.axes[0], 2e-4)


def resetBounds(ax, xmax, xmin=None):
    if xmin is None:
        xmin = -xmax

    tix = np.linspace(xmin, xmax, 3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xticks(tix)
    ax.set_yticks(tix)


class ThresholdSim(xc.Simulation):
    def __init__(self, name, xdom, dualElectrode, elecY=None, sigma=1., viz=None):
        bbox = xdom*np.concatenate((-np.ones(3), np.ones(3)))
        super().__init__(name, bbox)
        self.sigma = sigma

        if elecY is None:
            elecY = xdom/2

        elec0 = np.zeros(3)
        elec0[1] = elecY*1e-6
        elecA = elec0.copy()
        elecB = elec0.copy()

        if dualElectrode:
            elecA[0] = 2*rElec
            elecB[0] = -2*rElec
            wire2 = xc.geometry.Disk(elecB,
                                     rElec,
                                     np.array([0, 0, 1]))
            self.addCurrentSource(value=-1.,
                                  coords=elecB,
                                  geometry=wire2)

            if viz is not None:
                xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                                  srcCenter=elecB[:-1])

        if viz is not None:
            xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                              srcCenter=elecA[:-1])

        wire1 = xc.geometry.Disk(elecA,
                                 rElec,
                                 np.array([0, 0, 1], dtype=float))

        self.addCurrentSource(value=1.,
                              coords=elecA,
                              geometry=wire1)

    def meshAndSolve(self, depth):
        metrics = []
        for src in self.currentSources:
            metrics.append(xc.makeExplicitLinearMetric(maxdepth=depth,
                                                       meshdensity=0.2,
                                                       origin=src.geometry.center))

        self.makeAdaptiveGrid(metrics, depth)

        self.finalizeMesh()
        self.setBoundaryNodes()
        v = self.iterativeSolve(tol=1e-9)

    def getAnalyticVals(self, coords):
        anaVals = np.zeros(coords.shape[0])
        for src in self.currentSources:
            anaVals += xc.util.pointCurrentV(coords, iSrc=src.value,
                                             sigma=self.sigma,
                                             srcLoc=src.coords)

        return anaVals


class ThresholdStudy:
    def __init__(self, simulation, pulsedur=1., biphasic=True, viz=None):
        self.segCoords = None
        self.vExt = None
        self.sim = simulation
        self.viz = viz
        self.isBiphasic = biphasic
        self.pulsedur = pulsedur

    def _buildNeuron(self):
        cell = com.BallAndStick(1, -50., 0., 0., 0.)
        cell.dend.nseg = 15
        h.define_shape()

        h.nlayer_extracellular(1)
        cellcoords = []
        inds = []
        for nsec, sec in enumerate(h.allsec()):
            sec.insert('extracellular')

            ptN = sec.n3d()

            for ii in range(ptN):
                cellcoords.append([sec.x3d(ii), sec.y3d(ii), sec.z3d(ii)])
                inds.append([[nsec, ii]])

        self.segCoords = np.array(cellcoords)/nUnit.m

        # optional visualization
        if self.viz is not None:
            # viz.addSimulationData(setup,append=True)
            xc.nrnutil.showCellGeo(self.viz.axes[0])

        return cell

    def getAnalyticThreshold(self, pmin=1e-6, pmax=1e-2):

        assert self._runTrial(pmax, analytic=True)
        assert not self._runTrial(pmin, analytic=True)

        while (pmax-pmin) > 1e-6:
            md = 0.5*(pmin+pmax)
            # print(md)
            spike = self._runTrial(md, analytic=True)

            if spike:
                pmax = md
            else:
                pmin = md

        thresh = md
        return thresh

    def getThreshold(self, depth, pmin=1e-6, pmax=1e-2):
        self.sim.meshAndSolve(depth)
        if self.viz is not None:
            self.viz.addSimulationData(self.sim, append=True)
        numEl = len(self.sim.mesh.elements)
        numSrc = sum(self.sim.nodeRoleTable == 2)

        if numSrc == 0:
            numEl = np.nan
            thresh = np.nan
            numSrc = np.nan
        else:
            assert self._runTrial(pmax)
            assert not self._runTrial(pmin)

            while (pmax-pmin) > 1e-6:
                md = 0.5*(pmin+pmax)
                # print(md)
                spike = self._runTrial(md)

                if spike:
                    pmax = md
                else:
                    pmin = md

            thresh = md
            # in amps

        return thresh, numEl, numSrc

    def _runTrial(self, amplitude, analytic=False):
        cell = self._buildNeuron()

        if self.isBiphasic:
            tstim, vstim = xc.nrnutil.makeBiphasicPulse(
                amplitude, 2., self.pulsedur)
        else:
            tstim, vstim = xc.nrnutil.makeMonophasicPulse(
                amplitude, 2., self.pulsedur)

        self.tstim = tstim
        self.vstim = vstim
        tstop = 10*max(tstim.as_numpy())

        vvecs, vmems = self._setVext(analytic=analytic)

        tvec = h.Vector().record(h._ref_t)

        h.finitialize(-65*nUnit.mV)

        h.continuerun(tstop)

        memVals = cell.soma_v.as_numpy()
        t = tvec.as_numpy()

        spiked = np.any(memVals > 0)

        del cell

        for sec in h.allsec():
            h.delete_section(sec=sec)

        # # for vec in h.allobjects('Vector'):
        # #     vec.play_remove()
        tvec.play_remove()
        tstim.play_remove()
        vstim.play_remove()
        # [v.play_remove() for v in vvecs]
        # [v.play_remove() for v in vmems]

        return spiked

    def _setVext(self, analytic=False):

        setup = self.sim
        if self.vExt is None:
            if analytic:
                vext = self.sim.getAnalyticVals(self.segCoords)
            else:
                vext = setup.interpolateAt(self.segCoords)
            self.vExt = vext
        else:
            vext = self.vExt

        # tstim,vstim =xc.nrnutil.makeBiphasicPulse(k, tstart, tpulse)

        vvecs = []
        vmems = []
        for sec, V in zip(h.allsec(), vext):
            for seg in sec.allseg():
                vseg = V*nUnit.V

                vvecs.append(h.Vector(vseg*self.vstim.as_numpy()))
                vvecs[-1].play(seg.extracellular._ref_e, self.tstim, False)
                vmems.append(h.Vector().record(seg._ref_v))
                # vvecs[-1].play(seg._ref_e_extracellular, tstim, False)

        return vvecs, vmems


h.load_file('stdrun.hoc')


if analytic:

    y = np.linspace(0, 200)
    threshSet = []

    for d in y:
        sim = ThresholdSim('analytic', domX, elecY=elecY,
                           dualElectrode=pairStim,
                           sigma=Sigma)
        tst = ThresholdStudy(sim)
        thresh = tst.getAnalyticThreshold()
        threshSet.append(thresh)

        del tst
        # print(thresh)

    _, ax = plt.subplots()
    ax.plot(y, threshSet)
    xc.visualizers.engineerTicks(ax, 'm', 'A')


else:
    threshSet = []
    nElec = []
    nTot = []

    sim = ThresholdSim('', domX, elecY=elecY, dualElectrode=pairStim, viz=viz)
    tmp = ThresholdStudy(sim, pulsedur=tpulse)
    threshAna = tmp.getAnalyticThreshold()

    for d in range(3, 14):
        tst = ThresholdStudy(sim, viz=viz)

        t, nT, nE = tst.getThreshold(d)
        del tst

        threshSet.append(t)
        nElec.append(nE/len(sim.currentSources))
        nTot.append(nT)

        if nE > 1000:
            break

    if pairStim:
        stimStr = 'bi'

    else:
        stimStr = 'mono'

    f, ax = plt.subplots()
    simline, = ax.semilogx(nElec, threshSet, marker='o', label='Simulated')
    ax.set_xlabel('Source nodes')

    ax.set_ylabel('Threshold')
    ax.yaxis.set_major_formatter(xc.visualizers.eform('A'))

    ax.hlines(threshAna, 1, np.nanmax(nElec), label='Analytic value', linestyles=':',
              colors=simline.get_color())
    ax.legend()

    ax.set_title(r'Activation threshold for %.1f ms %sphasic pulse %d $\mu m$ away' % (
        tpulse, stimStr, elecY))

    if save:
        dset = {'nElec': nElec, 'nTot': nTot,
                'thresh': threshSet, 'threshAna': threshAna}
        fstub = study.studyPath+'/'
        pickle.dump(dset, open(fstub+'res.dat', 'wb'))
        aniName = fstub+'steps'

        study.savePlot(f, 'thresh')

    else:
        aniName = None

    if stepViz:
        ani = viz.animateStudy(aniName)


combineRuns = True
if combineRuns:
    import os
    bpath, _ = os.path.split(study.studyPath)
    folders = os.listdir(bpath)
    Ys = [q.split('y')[1] for q in folders]
    dx = [pickle.load(open(os.path.join(bpath, fl, 'res.dat'), 'rb'))
          for fl in folders]

    yvals = [int(y) for y in Ys]
    order = np.argsort(yvals)

    f, ax = plt.subplots()
    ax.set_xlabel('Source nodes')

    ax.set_ylabel('Threshold')
    ax.yaxis.set_major_formatter(xc.visualizers.eform('A'))

    for d, y in zip(np.array(dx)[order], np.array(Ys)[order]):
        simline, = ax.semilogx(d['nElec'], d['thresh'], marker='o', label=y)
        ax.hlines(d['threshAna'],
                  np.nanmin(d['nElec']),
                  np.nanmax(d['nElec']),
                  linestyles=':',
                  colors=simline.get_color())

    ax.legend()
    ax.set_title(r'Activation threshold for %.1f ms %sphasic pulse' %
                 (tpulse, stimStr))

    study.savePlot(f, 'composite')
