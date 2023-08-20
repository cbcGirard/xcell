#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:00:57 2022

Based on simulation protocol of Reilly 2016, using the Carnavale biophysics

@author: benoit
"""

import xcell as xc
import numpy as np
import Common_nongallery as com
from neuron import h  # , gui
import neuron.units as nUnit

import matplotlib.pyplot as plt
import pickle


xvals=np.array([0,1,50])*1e-2
yvals=np.array([.25, 1])*1e-2
pulseTvals=np.array([0.005, 2.0])*nUnit.ms

pulseSet=pulseTvals[[0,1,0,1,1,1,0,1]]
xAset=xvals[[2,2,2,2,0,0,2,2]]
yAset=yvals[[0,0,0,0,0,1,0,0]]
xCset=xvals[[0,0,0,0,2,1,0,0]]
yCset=yvals[[0,0,1,1,0,1,0,0]]
biphase=[0,0,0,0,0,0,1,1]



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
# 300 ohm-cm
Sigma = 1/3 #S/m



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
                                      'show_nodes': True,
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
            self.add_current_source(value=-1.,
                                  geometry=wire2)

            if viz is not None:
                xc.visualizers.show_source_boundary(viz.axes, rElec,
                                                  source_center=elecB[:-1])

        if viz is not None:
            xc.visualizers.show_source_boundary(viz.axes, rElec,
                                              source_center=elecA[:-1])

        wire1 = xc.geometry.Disk(elecA,
                                 rElec,
                                 np.array([0, 0, 1], dtype=float))

        self.add_current_source(value=1.,
                              geometry=wire1)

    def mesh_and_solve(self, depth):
        metrics = []
        for src in self.current_sources:
            metrics.append(xc.makeExplicitLinearMetric(max_depth=depth,
                                                       meshdensity=0.2,
                                                       origin=src.geometry.center))

        self.make_adaptive_grid(metrics, depth)

        self.finalize_mesh()
        self.set_boundary_nodes()
        v = self.solve(tol=1e-9)

    def get_analytic_vals(self, coords):
        analytic_values = np.zeros(coords.shape[0])
        for src in self.current_sources:
            analytic_values += xc.util.point_current_source_voltage(coords, i_source=src.value,
                                             sigma=self.sigma,
                                             source_location=src.coords)

        return analytic_values


class ThresholdStudy:
    def __init__(self, simulation, pulsedur=1., biphasic=True, viz=None):
        self.segment_coordinates = None
        self.v_external = None
        self.sim = simulation
        self.viz = viz
        self.is_biphasic = biphasic
        self.pulsedur = pulsedur

    def _build_neuron(self):
        h.load_file('estimsurvey/axon10.hoc')

        # cell = com.BallAndStick(1, -50., 0., 0., 0.)
        # cell.dend.nseg = 15
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

        self.segment_coordinates = np.array(cellcoords)/nUnit.m

        # optional visualization
        if self.viz is not None:
            # viz.add_simulation_data(setup,append=True)
            xc.nrnutil.show_cell_geo(self.viz.axes[0])

        resultVec=h.Vector().record(h.node[0](0.5)._ref_v)
        return resultVec
        # return cell

    def getAnalyticThreshold(self, pmin=1e-6, pmax=1e-2):

        assert self._run_trial(pmax, analytic=True)
        assert not self._run_trial(pmin, analytic=True)

        while (pmax-pmin) > 1e-6:
            md = 0.5*(pmin+pmax)
            # print(md)
            spike = self._run_trial(md, analytic=True)

            if spike:
                pmax = md
            else:
                pmin = md

        threshold = md
        return threshold

    def get_threshold(self, depth, pmin=1e-6, pmax=1e-2):
        self.sim.mesh_and_solve(depth)
        if self.viz is not None:
            self.viz.add_simulation_data(self.sim, append=True)
        n_elements = len(self.sim.mesh.elements)
        n_sources = sum(self.sim.node_role_table == 2)

        if n_sources == 0:
            n_elements = np.nan
            threshold = np.nan
            n_sources = np.nan
        else:
            assert self._run_trial(pmax)
            assert not self._run_trial(pmin)

            while (pmax-pmin) > 1e-6:
                md = 0.5*(pmin+pmax)
                # print(md)
                spike = self._run_trial(md)

                if spike:
                    pmax = md
                else:
                    pmin = md

            threshold = md
            # in amps

        return threshold, n_elements, n_sources

    def _run_trial(self, amplitude, analytic=False):
        nodeVec=self._build_neuron()

        if self.is_biphasic:
            tstim, vstim = xc.nrnutil.make_biphasic_pulse(
                amplitude, 2., self.pulsedur)
        else:
            tstim, vstim = xc.nrnutil.make_monophasic_pulse(
                amplitude, 2., self.pulsedur)

        self.tstim = tstim
        self.vstim = vstim
        tstop = 10*max(tstim.as_numpy())

        vvecs, vmems = self._set_v_ext(analytic=analytic)

        tvec = h.Vector().record(h._ref_t)

        h.finitialize(-65*nUnit.mV)

        h.continuerun(tstop)

        # memVals = cell.soma_v.as_numpy()
        memVals=nodeVec.as_numpy()
        t = tvec.as_numpy()

        spiked = np.any(memVals > 0)


        for sec in h.allsec():
            h.delete_section(sec=sec)

        # # for vec in h.allobjects('Vector'):
        # #     vec.play_remove()
        tvec.play_remove()
        tstim.play_remove()
        vstim.play_remove()
        nodeVec.play_remove()
        # [v.play_remove() for v in vvecs]
        # [v.play_remove() for v in vmems]

        return spiked

    def _set_v_ext(self, analytic=False):

        setup = self.sim
        if self.v_external is None:
            if analytic:
                vext = self.sim.get_analytic_vals(self.segment_coordinates)
            else:
                vext = setup.interpolate_at_points(self.segment_coordinates)
            self.v_external = vext
        else:
            vext = self.v_external

        # tstim,vstim =xc.nrnutil.make_biphasic_pulse(k, tstart, tpulse)

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


# %% run sims

h.load_file('stdrun.hoc')
h.nrn_load_dll('estimsurvey/x86_64/libnrnmech.so')


if analytic:

    y = np.linspace(0, 200)
    thresholdSet = []

    for d in y:
        sim = ThresholdSim('analytic', domX, elecY=elecY,
                           dualElectrode=pairStim,
                           sigma=Sigma)
        tst = ThresholdStudy(sim)
        threshold = tst.getAnalyticThreshold()
        thresholdSet.append(threshold)

        del tst
        # print(threshold)

    _, ax = plt.subplots()
    ax.plot(y, thresholdSet)
    xc.visualizers.engineering_ticks(ax, 'm', 'A')


else:
    thresholdSet = []
    nElec = []
    nTot = []

    sim = ThresholdSim('', domX, elecY=elecY, dualElectrode=pairStim, viz=viz)
    tmp = ThresholdStudy(sim, pulsedur=tpulse)
    thresholdAna = tmp.getAnalyticThreshold()

    for d in range(3, 14):
        tst = ThresholdStudy(sim, viz=viz)

        t, nT, nE = tst.get_threshold(d)
        del tst

        thresholdSet.append(t)
        nElec.append(nE/len(sim.current_sources))
        nTot.append(nT)

        if nE > 1000:
            break

    if pairStim:
        stimStr = 'bi'

    else:
        stimStr = 'mono'

    f, ax = plt.subplots()
    simline, = ax.semilogx(nElec, thresholdSet, marker='o', label='Simulated')
    ax.set_xlabel('Source nodes')

    ax.set_ylabel('Threshold')
    ax.yaxis.set_major_formatter(xc.visualizers.eform('A'))

    ax.hlines(thresholdAna, 1, np.nanmax(nElec), label='Analytic value', linestyles=':',
              colors=simline.get_color())
    ax.legend()

    ax.set_title(r'Activation thresholdold for %.1f ms %sphasic pulse %d $\mu m$ away' % (
        tpulse, stimStr, elecY))

    if save:
        dset = {'nElec': nElec, 'nTot': nTot,
                'threshold': thresholdSet, 'thresholdAna': thresholdAna}
        fstub = study.study_path+'/'
        pickle.dump(dset, open(fstub+'res.dat', 'wb'))
        aniName = fstub+'steps'

        study.save_plot(f, 'threshold')

    else:
        aniName = None

    if stepViz:
        ani = viz.animate_study(aniName)


combineRuns = True
if combineRuns:
    import os
    bpath, _ = os.path.split(study.study_path)
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
        simline, = ax.semilogx(d['nElec'], d['threshold'], marker='o', label=y)
        ax.hlines(d['thresholdAna'],
                  np.nanmin(d['nElec']),
                  np.nanmax(d['nElec']),
                  linestyles=':',
                  colors=simline.get_color())

    ax.legend()
    ax.set_title(r'Activation thresholdold for %.1f ms %sphasic pulse' %
                 (tpulse, stimStr))

    study.save_plot(f, 'composite')
