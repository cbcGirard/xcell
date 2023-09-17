#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFP estimation
=============================

Plot LFP from toy neurons

"""
import xcell as xc
from xcell import nrnutil as nUtil

from neuron import h  # , gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from tqdm import trange

import Common_nongallery
import time
import pickle
import os


import argparse

h.load_file("stdrun.hoc")
h.CVode().use_fast_imem(1)

mpl.style.use("fast")


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
cli.add_argument("-f", "--folder", help="path from main results folder", default="/tmp")
cli.add_argument(
    "-n", "--nRing", type=int, help="number of cells (default %(default)s) 0->single compartment", default=5
)
cli.add_argument("-s", "--nSegs", type=int, help="segments per dendrite (default %(default)s", default=5)
cli.add_argument("-k", "--nskip", type=int, help="timesteps to skip", default=6)
cli.add_argument(
    "--strat", help="meshing approach", choices=["fixedMin", "depth", "fixedMax", "depthExt"], default="fixedMin"
)
cli.add_argument("-v", "--vids", help="generate animation", action="store_true")
cli.add_argument("-p", "--post", help="generate summary plots", action="store_true")
cli.add_argument("-S", "--synth", help="use dummy pulses instead of biopyhsical sources", action="store_true")


args = cli.parse_args()

# Overrides for e.g. debugging in Spyder
args.vids = True
# args.synth=False
args.folder = "Final/polyCell"
# args.folder='Quals/monoCell'
args.strat = "depth"
# args.nSegs = 101
# args.folder='tst'
# args.nskip=1
# args.nRing=0
xc.colors.use_light_style()

# %%
if args.strat == "depthExt":
    dmax += 4

if args.nRing == 0:
    ring = Common_nongallery.Ring(N=1, stim_delay=0, dendSegs=1, r=0)
    tstop = 12
    tPerFrame = 2
    barRatio = [4, 1]
else:
    ring = Common_nongallery.Ring(N=args.nRing, stim_delay=0, dendSegs=args.nSegs, r=175)
    tstop = 40
    tPerFrame = 5
    barRatio = [9, 1]

coords, rads, is_sphere = nUtil.get_neuron_geometry()
ivecs = nUtil.get_membrane_currents()

if args.nRing == 0:
    ivecs.pop()
    coords.pop()
    rads.pop()

if args.synth:
    ncomp = len(rads)
    tv = np.linspace(0, 4 + 2 * ncomp)

    def synthPulse(pulseT):
        posPhase = np.exp(-10 * (tv - 1.5 * pulseT - 2) ** 2)
        negPhase = np.exp(-10 * (tv - 1.5 * pulseT - 3) ** 2)

        return 1e-9 * (posPhase - negPhase)

    I = np.array([synthPulse(k) for k in range(ncomp)]).transpose()
else:
    t = h.Vector().record(h._ref_t)
    t.append(tstop)
    h.finitialize(-65 * mV)
    h.continuerun(tstop)

    tv = np.array(t) * 1e-3
    I = 1e-9 * np.array(ivecs).transpose()

which_pts = list(range(0, tv.shape[0], args.nskip))
which_pts.append(tv.shape[0] - 1)
whichInds = np.array(which_pts)

I = I[whichInds]
tv = tv[whichInds]


analyticVmax = I / (4 * np.pi * np.array(rads, ndmin=2))
vPeak = np.max(np.abs(analyticVmax))

print("Vrange\n%.2g\t%.2g\n" % (np.min(analyticVmax), np.max(analyticVmax)))
imax = np.max(np.abs(I[:: args.nskip]))


coord = np.array(coords)
xmax = 2 * np.max(np.concatenate((np.max(coord, axis=0), np.min(coord, axis=0))))

# round up
xmax = xc.util.round_to_digit(xmax)
if xmax <= 0 or np.isnan(xmax):
    xmax = 1e-4


lastNumEl = 0


resultFolder = os.path.join(args.folder, args.strat)

study, setup = Common_nongallery.makeSynthStudy(resultFolder, xmax=xmax)
setup.current_sources = []
study_path = study.study_path

dspan = dmax - dmin


if args.nRing == 0:
    tdata = {
        "x": tv,
        "y": I,
        "style": "sweep",
        "ylabel": "",
        "unit": "A",
    }
else:
    tdata = None

if args.vids:
    img = xc.visualizers.SingleSlice(None, study, tv, tdata=tdata)

    err = xc.visualizers.SingleSlice(None, study, tv, tdata=tdata)
    err.dataSrc = "absErr"
    # err.dataSrc='vAna'

    animators = [img, err]
    aniNames = ["volt-", "error-"]

    # animators = [img]
    # aniNames = ['volt-']

    # animators=[xc.visualizers.ErrorGraph(None, study)]
    # animators.append(xc.visualizers.LogError(None,study))

iEffective = []
for r, c, v in zip(rads, coords, I.transpose()):
    signal = xc.signals.PiecewiseSignal()
    signal.times = tv
    signal.values = v
    geo = xc.geometry.Sphere(c, r)
    setup.add_current_source(signal, geo)

tmax = tv.shape[0]
errdicts = []

polygons = nUtil.get_cell_image()

simDict = {
    "tv": tv,
    "I": I,
    "rads": rads,
    "coords": coords,
    "is_sphere": is_sphere,
    "polygons": polygons,
}
pickle.dump(simDict, open(os.path.join(os.path.dirname(study_path), "commonData.xstudy"), "wb"))


# %%
# run simulatios
stepper = trange(0, tmax)
for ii in stepper:
    t0 = time.monotonic()
    ivals = I[ii]
    tval = tv[ii]
    vScale = np.abs(analyticVmax[ii]) / vPeak

    setup.current_time = tval

    changed = False
    # metrics=[]

    # for jj in range(len(setup.current_sources)):
    #     ival = ivals[jj]
    #     setup.current_sources[jj].value = ival
    setup.current_time = tval

    if args.strat == "k":
        # k-param strategy
        density = 0.4 * vScale
        # print('density:%.2g'%density)

    elif args.strat[:5] == "depth" or args.strat == "d2":
        # Depth strategy
        scale = dmin + dspan * vScale
        # dint, dfrac = divmod(np.min(scale), 1)
        dint = np.rint(scale)
        max_depth = np.floor(scale).astype(int)
        # print('depth:%d'%max_depth)

        # density=0.25
        if args.strat == "d2":
            density = 0.5
        else:
            density = 0.2  # +0.2*dfrac

        metricCoef = 2 ** (-density * scale)
        # metricCoef=2**(-density*dmax*vScale)

    elif args.strat[:5] == "fixed":
        # Static mesh
        density = 0.2
        if args.strat == "fixedMax":
            max_depth = dmax * np.ones_like(vScale)
            dmin = dmax
        elif args.strat == "fixedMin":
            max_depth = dmin * np.ones_like(vScale)
            dmax = dmin
        else:
            max_depth = 5
        metricCoef = 2 ** (-max_depth * density)

    netScale = 2 ** (-max_depth * density)

    changed = setup.make_adaptive_grid(coord, max_depth, xc.general_metric, coefs=metricCoef)

    if changed or ii == 0:
        setup.meshnum += 1
        setup.finalize_mesh()

        n_elements = len(setup.mesh.elements)

        setup.set_boundary_nodes()

        v = setup.solve()
        lastNumEl = n_elements
        setup.iteration += 1

        study.save_simulation(setup)  # ,baseName=str(setup.iteration))
        stepper.set_postfix_str("%d source nodes" % sum(setup.node_role_table == 2))
    else:
        # vdof = setup.get_voltage_at_dof()
        # v=setup.solve(vGuess=vdof)
        # TODO: vguess slows things down?

        setup.node_role_table[setup.node_role_table == 2] = 0

        v = setup.solve()

    dt = time.monotonic() - t0

    study.log_current_simulation(["Timestep", "Meshnum"], [setup.current_time, setup.meshnum])

    setup.step_logs = []

    errdict = xc.misc.get_error_estimates(setup)
    errdict["densities"] = density
    errdict["depths"] = max_depth
    errdict["numels"] = lastNumEl
    errdict["dt"] = dt
    errdict["vMax"] = max(v)
    errdict["vMin"] = min(v)
    errdicts.append(errdict)

    iEffective.append(setup._node_current_sources)

    if args.vids:
        [an.add_simulation_data(setup, append=True) for an in animators]
        # err.add_simulation_data(setup,append=True)
        # img.add_simulation_data(setup, append=True)

lists = xc.misc.transpose_dicts(errdicts)

if args.strat[:5] == "fixed":
    if args.strat[5:] == "Min":
        dstr = dmin
    else:
        dstr = dmax
    depthStr = "Fixed depth %d" % dstr
else:
    depthStr = "Dynamic depth [%d:%d]" % (dmin, dmax)


lists["depths"] = depthStr

# pickle.dump(lists, open(study_path+'/'+args.strat+'.pcr', 'wb'))
study.save(lists, args.strat)
if args.vids:
    for ani, aniName in zip(animators, aniNames):
        study.save_animation(ani, aniName)

# %%

if args.vids:
    # for lite in ['','-lite']:
    #     if lite=='':
    #         xc.colors.use_light_style()
    #     else:
    #         xc.colors.use_dark_style()
    #         anis=[an.animate_study(l+args.strat,fps=30) for an,l in zip(animators, aniNames)]

    for an, l in zip(animators, aniNames):
        xc.colors.use_dark_style()
        nUtil.show_cell_geo(an.axes[0])

        if len(an.datasets) == 0:
            # reload data
            an = study.load(l, ext=".adata")

        fname = l + args.strat
        # an.animate_study(fname, fps=30)

        xc.colors.use_light_style()

        alite = an.copy({"colorbar": False, "barRatio": barRatio, "labelAxes": False})

        figw = 0.3 * 7
        alite.fig.set_figwidth(figw)
        alite.fig.set_figheight(1.1 * figw)
        alite.axes[0].set_xticks([])
        alite.axes[0].set_yticks([])

        nUtil.show_cell_geo(alite.axes[0])

        # get closest frames to 5ms intervals
        frameNs = np.linspace(0, len(alite.datasets) - 1, 1 + int(tstop / tPerFrame), dtype=int)
        artists = [alite.get_artists(ii) for ii in frameNs]
        alite.animate_study(
            fname + "-lite", fps=30, artists=artists, vector_frames=np.arange(len(frameNs)), unitStr="V"
        )
        alite.solobar(fname + "-lite")

        # artists=[alite.get_artists(ii) for ii in frameNs]
        # alite.animate_study(fname+'-lite', fps=30, vector_frames=frameNs, unitStr='V')

# %%
if args.post:
    folderstem = os.path.dirname(study_path)
    nuStudy, _ = Common_nongallery.makeSynthStudy(folderstem)

    def getdata(strat):
        fname = os.path.join(folderstem, strat, strat + ".pcr")
        data = pickle.load(open(fname, "rb"))
        return data

    labels = [l for l in os.listdir(folderstem) if os.path.isdir(os.path.join(folderstem, l))]
    labels.sort()
    # labels = [labels[n] for n in [3, 0, 2, 1]]
    labels = [labels[n] for n in [3, 0, 2]]
    data = [getdata(l) for l in labels]

    with mpl.rc_context(
        {
            "lines.markersize": 2.5,
            "lines.linewidth": 1,
            "figure.figsize": [3.25, 2.5],
            "font.size": 10,
            "legend.fontsize": 9,
        }
    ):
        for lite in ["", "-lite"]:
            if lite == "":
                xc.colors.use_dark_style()
            else:
                xc.colors.use_light_style()

            f, ax = plt.subplots(3, gridspec_kw={"height_ratios": [5, 5, 2]}, figsize=[3.25, 3.5])
            ax[2].plot(tv, I)

            for d, l in zip(data, labels):
                # ax[0].semilogy(tv, np.abs(d['volErr']), label=d['depths'])
                ax[1].plot(tv, np.abs(d["intErr"]), label=d["depths"])
                ax[0].plot(tv[1:], d["dt"][1:], label=d["depths"])

            ax[0].legend(loc="right")
            ax[1].set_ylabel("Error")
            ax[0].set_ylabel("Wall time [s]")
            ax[2].set_ylabel("Activity")
            ax[2].set_xlabel("Simulated time")
            ax[2].set_xlim(0, tstop * 1e-3)

            xc.visualizers.engineering_ticks(ax[2], xunit="s")
            [a.set_xticks([]) for a in ax[:-1]]
            f.align_ylabels()

            f2, ax = plt.subplots(1, 1)
            aright = ax.twinx()

            # [a.set_yscale('log') for a in [ax,aright]]

            ttime = [sum(l["dt"]) for l in data]
            terr = [sum(np.abs(l["intErr"])) for l in data]
            # terr=[sum(np.abs(l['intErr']))/sum(np.abs(l['intAna'])) for l in data]
            # terr=[sum(np.abs(l['SSE']))/sum(np.abs(l['SSTot'])) for l in data]

            categories = [d["depths"] for d in data]
            # tweak labels
            categories = [c.replace("depth ", "\n[") + "]" for c in categories]
            categories = [c.replace("[[", "[").replace("]]", "]") for c in categories]

            barpos = np.arange(len(categories))
            barw = 0.25

            tart = ax.bar(barpos - barw / 2, height=ttime, width=barw, color="C0", label="Time")
            ax.set_ylabel("Total simulation time [s]")

            eart = aright.bar(barpos + barw / 2, height=terr, width=barw, color="C8", label="Error")
            aright.set_ylabel("Total simulation error")

            ax.set_xticks(barpos)
            ax.set_xticklabels(categories)

            ax.legend(handles=[tart, eart], loc="upper center")

            axes = [ax, aright]
            ntick = 0

            nticks = [len(a.get_yticks()) for a in axes]
            ntick = max(nticks)
            for a in axes:
                dtick = a.get_yticks()[1]
                a.set_yticks(np.arange(ntick) * dtick)

            study.save_plot(f2, os.path.join(folderstem, "ring%dsummary%s" % (args.nRing, lite)))

            study.save_plot(f, os.path.join(folderstem, "ring%dcomparison%s" % (args.nRing, lite)))


# #%% Semi- manual relabling
# for l in labels:
#     d=getdata(l)
#     if l[:5]=='fixed':
#         if l[5:]=='Min':
#             dstr=dmin
#         else:
#             dstr=dmax
#         s='Fixed depth %d'%dstr
#     else:
#         if len(l)>5:
#             dmax=12
#         else:
#             dmax=8
#         s='Dynamic depth [%d:%d]'%(dmin,dmax)
#     d['depths']=s
#     pickle.dump(d,open(os.path.join(folderstem,l,l+'.pcr'), 'wb'))
