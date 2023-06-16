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
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr
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
parser.add_argument('-C', '--celltype', help='',
                    choices=['ball', 'MRG', 'Ax10'], default='ball')

# domX = 2.50e-2

save = True

pairStim = False
if pairStim:
    stimStr = 'bi'

else:
    stimStr = 'mono'


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
celltype = args.celltype

# # #overrides
# # domX = rElec*2**7
# # tpulse=0.1
# stepViz = True
celltype = 'MRG'
args.folder = 'tthreshold-%s' % celltype
combineRuns = True
args.liteMode = True
# elecY = 10
# # analytic = True


def resetBounds(ax, xmax, xmin=None, yElec=0):
    if xmin is None:
        xmin = -xmax

    tix = np.linspace(xmin, xmax, 3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin-yElec, xmax-yElec)
    ax.set_xticks(tix)
    ax.set_yticks(tix)


class ThisSim(xc.nrnutil.ThresholdSim):
    def __init__(self, name, xdom, dualElectrode, elecY=None, sigma=1., viz=None):
        bbox = xdom*np.concatenate((-np.ones(3), np.ones(3)))
        self.sigma = sigma

        if elecY is None:
            elecY = xdom/2

        elec0 = np.zeros(3)
        # elec0[1] = elecY*1e-6
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
            # anaVals += xc.util.diskCurrentV(coords,
            # anaVals += xc.util.pointCurrentV(coords,
            #                                  iSrc=src.value.getValueAtTime(0.),
            #                                  sigma=self.sigma,
            #                                  srcLoc=src.coords)
            dists = src.geometry.getSignedDistance(coords)
            k = src.value.getValueAtTime(0.)/(
                4*np.pi*self.sigma)
            vals = k/dists
            vals[dists < src.geometry.radius] = k/src.geometry.radius
            anaVals += vals

        return anaVals


class ThisStudy(xc.nrnutil.ThresholdStudy):
    def __init__(self, simulation, yval=0., pulsedur=1., biphasic=True, viz=None):
        self.yval = yval
        super().__init__(simulation=simulation,
                         pulsedur=pulsedur,
                         biphasic=biphasic,
                         viz=viz)

    def _buildNeuron(self):
        # h.load_file('stdrun.hoc')

        # cell = com.BallAndStick(1, -50., 0., 0., 0.)
        # cell.dend.nseg = 15

        domx = self.sim.mesh.bbox[3]

        nnodes = 9
        segL = 10.

        if celltype == 'Ax10':
            cell = com.Axon10(1, -(nnodes//2)*segL,
                              -self.yval, 0, nnodes,
                              segL=segL)
        if celltype == 'MRG':
            cell = com.MRG(1, -(nnodes//2)*1150, -self.yval,
                           0., 0., axonNodes=nnodes)
        else:
            cell = com.BallAndStick(1, -100, -self.yval, 0, 0)

        self.segCoords = xc.nrnutil.makeInterface()

        # optional visualization
        if self.viz is not None:
            # viz.addSimulationData(setup,append=True)
            self.cellImg = xc.nrnutil.showCellGeo(
                self.viz.axes[0], showNodes=False)

        return cell


# xview = 3*np.max([elecY*1e-6, rElec])
xview = 4e-4

if analytic:
    stepViz = False

if args.liteMode:
    xc.colors.useLightStyle()
else:
    xc.colors.useDarkStyle()


study, _ = com.makeSynthStudy(os.path.join(
    args.folder, 'y'+str(elecY)), xmax=domX)


if stepViz:
    viz = xc.visualizers.SliceSet(None, study,
                                  prefs={
                                      'showError': False,
                                      'showInsets': False,
                                      'relativeError': False,
                                      'logScale': True,
                                      'showNodes': True,
                                      'fullInterp': True})
    resetBounds(viz.axes[0], xview)  # , yElec=elecY*1e-6)
else:
    viz = None
    # zoom in
    # resetBounds(viz.axes[0], 2e-4)

# %%


if __name__ == '__main__':

    if analytic:

        threshSet = []

        sim = ThisSim('analytic', domX, elecY=elecY,
                      dualElectrode=pairStim,
                      sigma=Sigma, viz=viz)
        tst = ThisStudy(sim)
        thresh = tst.getThreshold(None, analytic=analytic)
        threshSet.append(thresh)

        # print(thresh)

        _, ax = plt.subplots()
        ax.plot(y, threshSet)
        xc.visualizers.engineerTicks(ax, 'm', 'A')

    else:
        threshSet = []
        nElec = []
        nTot = []
        l0ratio = []
        dataset = []

        sim = ThisSim('', domX, elecY=elecY,
                      dualElectrode=pairStim, viz=viz,
                      sigma=Sigma)
        tmp = ThisStudy(sim, pulsedur=tpulse, yval=elecY, viz=viz)
        threshAna, _, _ = tmp.getThreshold(0, analytic=True)
        del tmp

        # depthvec = trange(3, 14, desc='Simulating')
        # # for d in range(3, 14):
        # for d in depthvec:
        d = 7
        rcenter = 2**d
        # xvec = 48e-6*np.geomspace(10*rcenter, rcenter/10)
        # xstepper = trange(len(xvec))

        if pairStim:
            l0target = rElec*2.**(1-np.arange(8))
        else:
            l0target = rElec*np.geomspace(10, 0.1)  # , 20)
        xstepper = trange(len(l0target))

        for ii in xstepper:
            l0targ = l0target[ii]
            d = int(np.ceil(np.log2(domX/l0targ)))
            xdom = l0targ*2**d

            sim = ThisSim('', xdom, elecY=elecY,
                          dualElectrode=pairStim,
                          viz=viz, sigma=Sigma)
            tst = ThisStudy(sim, viz=viz, yval=elecY)

            t, nT, nE = tst.getThreshold(
                d, pmin=0, pmax=threshAna*1e2, strict=True)
            if np.isnan(t):
                continue

            minl0 = min([el.l0 for el in tst.sim.mesh.elements])
            l0ratio.append(minl0/rElec)

            data = {}
            data['threshold'] = t
            data['nElec'] = nE/len(sim.currentSources)
            data['nTot'] = nT
            data['l0ratio'] = minl0/rElec
            data['xcoords'] = tst.segCoords[:, 0].squeeze()
            data['segV'] = tst.vExt
            data['l0min'] = minl0
            data['threshAna'] = threshAna

            dataset.append(data)

            xstepper.set_postfix_str('threshold: %.1g' % t)

            del tst

            threshSet.append(t)
            nElec.append(nE/len(sim.currentSources))
            nTot.append(nT)

            if nE > 5e3:
                break

        dataVecs = xc.misc.transposeDicts(dataset)

        f, ax = plt.subplots()
        # simline, = ax.semilogx(nElec, threshSet,
        # simline, = ax.semilogx(l0ratio, threshSet,
        #                        marker='o', label='Simulated')
        simline, = ax.semilogx(dataVecs['l0ratio'],
                               dataVecs['threshold'],
                               marker='o', label='Simulated')
        # ax.set_xlabel('Source nodes')
        ax.set_xlabel(r'$\ell_0/r_{elec}$')
        ax.xaxis.set_inverted(True)

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

        # voltage field
        f2, axes2 = plt.subplots(nrows=2, sharex=True, layout='constrained')
        # [a.set_yscale('log') for a in axes2]
        axes2[0].set_ylabel('Extracellular V @ 1A')
        axes2[1].set_ylabel('Extracellular V @ threshold')
        axes2[1].xaxis.set_major_formatter(xc.visualizers.eform('m'))

        nsims = len(dataset)

        cmap = plt.get_cmap('viridis_r', nsims)
        spec = np.flipud(cmap.colors)
        # spec = cmr.take_cmap_colors('cmr.chroma_r', N=nsims)

        for ii in range(nsims):
            dat = dataset[ii]

            yvals = [dat['segV'], dat['threshold']*dat['segV']]
            for a, y in zip(axes2, yvals):
                a.plot(dat['xcoords'],
                       y,
                       color=spec[ii],
                       marker='*',
                       )

        # cnorm = plt.Normalize(#vmin=0,
        cnorm = LogNorm(
            vmin=dataVecs['l0min'][-1],
            vmax=dataVecs['l0min'][0])
        cb = f2.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                               norm=cnorm),
                         ax=axes2,
                         format=xc.visualizers.eform('m'),
                         shrink=0.5)
        cb.set_label(r'$\ell_0$', loc='top')
        f2.align_ylabels()

        if save:
            # dset = {'nElec': nElec, 'nTot': nTot,
            # 'thresh': threshSet, 'threshAna': threshAna}
            fstub = study.studyPath+'/'
            # pickle.dump(dset,
            pickle.dump(dataVecs,
                        open(fstub+'res.dat', 'wb'))
            aniName = fstub+'steps'

            study.savePlot(f, 'thresh')

            study.savePlot(f2, 'vext')

        else:
            aniName = None

        if stepViz:
            ani = viz.animateStudy(aniName)

# %%
    if combineRuns:
        with plt.rc_context({
            'font.size': 10,
            'lines.markersize': 1.5,
            'lines.linewidth': 1,
            'figure.dpi': 300,
            'figure.figsize': [6.4, 3.6]
        }):
            # import os
            bpath, _ = os.path.split(study.studyPath)
            subs = os.listdir(bpath)
            folders = [f for f in subs
                       if os.path.isdir(os.path.join(bpath, f))]
            Ys = [q.split('y')[1] for q in folders]
            dx = [pickle.load(open(os.path.join(bpath, fl, 'res.dat'), 'rb'))
                  for fl in folders]

            yvals = [int(y) for y in Ys]
            order = np.argsort(yvals)

            f, ax = plt.subplots()
            f2, a2 = plt.subplots()

            for a in [ax, a2]:
                a.set_xlabel(r'$\ell_0/r_{elec}$')

                a.xaxis.set_inverted(True)

            ax.set_ylabel('Threshold')
            ax.yaxis.set_major_formatter(xc.visualizers.eform('A'))

            a2.set_ylabel(r'$i_{sim}/i_{analytic}$')

            # order = np.array([order[1], order[-1]])
            for d, y in zip(np.array(dx)[order], np.array(Ys)[order]):
                simline, = ax.semilogx(
                    # l0target/rElec,
                    d['l0ratio'],
                    # simline, = ax.loglog(
                    # d['nElec'],
                    d['threshold'],
                    marker='o', linestyle='-',
                    # label=r'$%s \mu m$, simulated' % y)
                    label=r'$%s \mu m$' % y)

                s2, = a2.loglog(d['l0ratio'],
                                np.array(d['threshold'])/d['threshAna'][0],
                                marker='o', linestyle='-',
                                label=r'$%s \mu m$' % y)

            xc.visualizers.outsideLegend(ax)
            xc.visualizers.outsideLegend(a2)

            ax.set_title(
                r'Activation threshold for %.1f ms biphasic pulse' % tpulse)
            a2.set_title('Threshold ratios ')
            t0, t1 = a2.get_ylim()
            a2.set_ylim(10**np.floor(np.log10(t0)),
                        10**np.ceil(np.log10(t1)))

            ymin, ymax = ax.get_ylim()
            ax.fill_betweenx(ax.get_ylim(), np.pi, 1., color='#99000080')
            txtpoint = min(0, ymin)
            txtpoint -= 0.025*(ymax-ymin)
            ax.text(np.sqrt(np.pi), txtpoint, 'Overshoot\nregion', horizontalalignment='center',
                    verticalalignment='top', color='#990000',
                    fontsize=8)

            ax.set_ylim(ymin, ymax)

            study.studyPath = bpath
            study.savePlot(f, 'composite')
            study.savePlot(f2, 'analyticComp')
