#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFP estimation with dynamic remeshing
===========================================

Plot LFP from toy neurons

"""

import xcell as xc
from neuron import h  # , gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange

# from ..Common_nongallery import Ring, makeSynthStudy

import Common_nongallery
import time

from xcell import nrnutil as nUtil

resultFolder = 'LFP'


h.load_file('stdrun.hoc')
h.CVode().use_fast_imem(1)

mpl.style.use('fast')

nRing = 5
nSegs = 5

dmax = 8
dmin = 3

nskip = 4

# %%

ring = Common_nongallery.Ring(N=nRing, stim_delay=0, dendSegs=nSegs, r=175)
tstop = 40
tPerFrame = 5
barRatio = [9, 1]

coords, rads, is_sphere = nUtil.get_neuron_geometry()
ivecs = nUtil.get_membrane_currents()


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

# round up
xmax = xc.util.round_to_digit(xmax)
if xmax <= 0 or np.isnan(xmax):
    xmax = 1e-4


lastNumEl = 0
lastI = 0

study, setup = Common_nongallery.makeSynthStudy(resultFolder, xmax=xmax)
setup.current_sources = []
study_path = study.study_path

dspan = dmax-dmin


tdata = None

img = xc.visualizers.SingleSlice(None, study,
                                    tv, tdata=tdata)

for r, c, i in zip(rads, coord, I.transpose()):
    geo = xc.geometry.Sphere(center=c, radius = r)
    signal=xc.signals.PiecewiseSignal(tv, i )
    setup.add_current_source(signal, geometry = geo)

tmax = tv.shape[0]
errdicts = []

stepper = trange(0,tmax,postfix="")

for ii in stepper:

    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]
    vScale = np.abs(analyticVmax[ii])/vPeak

    setup.current_time = tval

    changed = False

    # for jj in range(len(setup.current_sources)):
    #     ival = ivals[jj]
    #     setup.current_sources[jj].value = ival

    # Depth strategy
    scale = dmin+dspan*vScale
    dint = np.rint(scale)
    max_depth = np.floor(scale).astype(int)

    density = 0.2

    metricCoef = 2**(-density*scale)

    netScale = 2**(-max_depth*density)

    changed = setup.make_adaptive_grid(
        coord, max_depth, xc.general_metric, coefs=metricCoef)

    if changed or ii == 0:
        setup.meshnum += 1
        setup.finalize_mesh()

        n_elements = len(setup.mesh.elements)

        setup.set_boundary_nodes()

        v = setup.solve()
        lastNumEl = n_elements
        setup.iteration += 1

        study.save_simulation(setup)
        # stepper.set_postfix_str('%d source nodes' % sum(setup.node_role_table == 2))
        # stepper.set_postfix_str('%d elements'%n_elements)

        # stepper.set_postfix_str('(%.2g, %.2g)'%(v.min(), v.max()))
    else:
    #     # TODO: why is reset needed?
    #     setup.node_role_table[setup.node_role_table == 2] = 0
        v = setup.solve()

    dt = time.monotonic()-t0

    study.log_current_simulation(['Timestep', 'Meshnum'], [
                      setup.current_time, setup.meshnum])

    setup.step_logs = []


    img.add_simulation_data(setup, append=True)

nUtil.show_cell_geo(img.axes[0])

ani = img.animate_study('', fps=30)
