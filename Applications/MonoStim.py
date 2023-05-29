#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preliminary results for effect of mesh parameters on activation thresholds
"""

import xcell as xc
import numpy as np
import Common as com
from neuron import h  # , gui
import neuron.units as nUnit
from xcell.signals import Signal

import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument(
    '-Y', '--elecY', help='electrode distance in um', type=int, default=50)
parser.add_argument(
    '-x', '--xmax', help='span of domain in um', type=int, default=1000)
parser.add_argument(
    '-v', '--stepViz', help='generate animation of refinement', action='store_true')
parser.add_argument('-a', '--analytic', action='store_true',
                    help='use analyic point-sources instead of mesh')
parser.add_argument('-S', '--summary', action='store_true',
                    help='generate summary graph of results')

parser.add_argument('-f', '--folder', default='MonoStim')
parser.add_argument('-l', '--liteMode',
                    help='use light mode', action='store_true')

# domX = 2.50e-2

save = True

pairStim = False


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


args = parser.parse_args()

elecY = args.elecY
combineRuns = args.summary
stepViz = args.stepViz
analytic = args.analytic
domX = 1e-6*args.xmax/2


# #overrides
domX = rElec*2**7
# tpulse=0.1
stepViz = True

if analytic:
    stepViz = False

if args.liteMode:
    xc.colors.useLightStyle()
else:
    xc.colors.useDarkStyle()


study, _ = com.makeSynthStudy(
    args.folder+'y'+str(elecY), xmax=domX,)


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


class ThisSim(xc.nrnutil.ThresholdSim):
    def __init__(self, name, xdom, dualElectrode, elecY=None, sigma=1., viz=None):
        bbox = xdom*np.concatenate((-np.ones(3), np.ones(3)))
        self.sigma = sigma

        if elecY is None:
            elecY = xdom/2

        elec0 = np.zeros(3)
        elec0[1] = elecY*1e-6
        elecA = elec0.copy()
        elecB = elec0.copy()

        geoms = []
        amps = []

        if dualElectrode:
            elecA[0] = 2*rElec
            elecB[0] = -2*rElec
            wire2 = xc.geometry.Disk(elecB,
                                     rElec,
                                     np.array([0., 0., 1.]))

            geoms.append(wire2)
            amps.append(Signal(-1.))

            if viz is not None:
                xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                                  srcCenter=elecB[:-1])

        if viz is not None:
            xc.visualizers.showSourceBoundary(viz.axes, rElec,
                                              srcCenter=elecA[:-1])

        wire1 = xc.geometry.Disk(elecA,
                                 rElec,
                                 np.array([0., 0., 1.], dtype=float))

        geoms.append(wire1)
        amps.append(Signal(1.))

        super().__init__(name, xdom,
                         srcAmps=amps,
                         srcGeometry=geoms,
                         sigma=sigma)

    def getAnalyticVals(self, coords):
        anaVals = np.zeros(coords.shape[0])
        for src in self.currentSources:
            anaVals += xc.util.pointCurrentV(coords,
                                             iSrc=src.value.getValueAtTime(0.),
                                             sigma=self.sigma,
                                             srcLoc=src.coords)

        return anaVals


class ThisStudy(xc.nrnutil.ThresholdStudy):

    def _buildNeuron(self):
        # h.load_file('stdrun.hoc')

        # cell = com.BallAndStick(1, -50., 0., 0., 0.)
        # cell.dend.nseg = 15

        # nnodes = 21
        segL = 10.
        nnodes = 101

        # cell = com.Axon10(1, -(nnodes//2)*segL, 0, 0, nnodes, segL=segL)
        cell = com.MRG(1, -(nnodes//2)*1e3, 0., 0., 0., axonNodes=nnodes)

        self.segCoords = xc.nrnutil.makeInterface()

        # optional visualization
        if self.viz is not None:
            # viz.addSimulationData(setup,append=True)
            self.cellImg = xc.nrnutil.showCellGeo(
                self.viz.axes[0], showNodes=True)

        return cell

# %%


if __name__ == '__main__':

    if analytic:

        y = np.linspace(50, 200)
        threshSet = []

        for d in y:
            sim = ThisSim('analytic', domX, elecY=elecY,
                          dualElectrode=pairStim,
                          sigma=Sigma)
            tst = ThisStudy(sim)
            thresh = tst.getThreshold(None, analytic=analytic)
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
        l0ratio = []

        sim = ThisSim('', domX, elecY=elecY,
                      dualElectrode=pairStim, viz=viz)
        tmp = ThisStudy(sim, pulsedur=tpulse)
        threshAna, _, _ = tmp.getThreshold(0, analytic=True)
        del tmp

        # depthvec = trange(3, 14, desc='Simulating')
        # # for d in range(3, 14):
        # for d in depthvec:
        d = 7
        rcenter = 2**d
        xvec = 48e-6*np.geomspace(10*rcenter, rcenter/10)
        xstepper = trange(len(xvec))
        for xx in xstepper:
            xdom = xvec[xx]

            sim = ThisSim('', xdom, elecY=elecY,
                          dualElectrode=pairStim,
                          viz=viz)
            tst = ThisStudy(sim, viz=viz)

            t, nT, nE = tst.getThreshold(d, pmin=0, pmax=threshAna*1e2)
            minl0 = min([el.l0 for el in tst.sim.mesh.elements])
            l0ratio.append(minl0/rElec)
            plt.figure()
            plt.scatter(tst.segCoords[:, 0].squeeze(),
                        t*tst.vExt)

            del tst

            threshSet.append(t)
            nElec.append(nE/len(sim.currentSources))
            nTot.append(nT)

            if nE > 5e3:
                break

        if pairStim:
            stimStr = 'bi'

        else:
            stimStr = 'mono'

        f, ax = plt.subplots()
        # simline, = ax.semilogx(nElec, threshSet,
        simline, = ax.semilogx(l0ratio, threshSet,
                               marker='o', label='Simulated')
        ax.set_xlabel('Source nodes')

        ax.set_ylabel('Threshold')
        ax.yaxis.set_major_formatter(xc.visualizers.eform('A'))

        ax.hlines(threshAna,
                  # 1, np.nanmax(nElec),
                  l0ratio[0], l0ratio[-1],
                  label='Analytic value', linestyles=':',
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

    # combineRuns = True
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
            # simline, = ax.semilogx(
            simline, = ax.loglog(
                d['nElec'], d['thresh'], marker='o', linestyle=':', label=r'$%s \mu m$, simulated' % y)
            ax.hlines(d['threshAna'],
                      np.nanmin(d['nElec']),
                      np.nanmax(d['nElec']),
                      linestyles='-',
                      colors=simline.get_color(),
                      label=r'$%s \mu m$, analytic' % y)

        # ax.legend()
        xc.visualizers.outsideLegend(ax)
        ax.set_title(r'Activation threshold for %.1f ms %sphasic pulse' %
                     (tpulse, stimStr))

        study.savePlot(f, 'composite')
