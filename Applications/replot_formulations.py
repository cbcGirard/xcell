#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ad-hoc recreations of formulation data: closeup of errors vs. distance from source at gen 11

figure 7b in JNE paper
"""

import xcell as xc
import Common_nongallery as com
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


folder = "Quals/formulations"

nthGen = 11
DPI = 200
figdims = [3.25, 3]
npx = int(figdims[0] * DPI)

study, _ = com.makeSynthStudy(folder)
folder = study.study_path
xc.colors.use_light_style()


formulations = ["Admittance", "FEM", "Face"]
titles = ["Admittance", "Trilinear FEM", "Mesh dual"]

rs = []
vs = []


def compress(x, y, nPx, xmin, xmax):
    pxVec = np.geomspace(xmin, xmax, nPx)

    xcomp = []
    ycomp = []
    for ii in range(pxVec.shape[0]):
        if ii:
            p0 = pxVec[ii - 1]
        else:
            p0 = 0

        isin = np.logical_and(x >= p0, x <= pxVec[ii])

        if xc.util.fastcount(isin):
            minval = np.min(y[isin])
            midval = np.mean(y[isin])
            maxval = np.max(y[isin])

            ycomp.extend([minval, midval, maxval])
            xcomp.extend(3 * [pxVec[ii]])

    return xcomp, ycomp


for form, title in zip(formulations, titles):
    graph = pickle.load(open(os.path.join(folder, "ErrorGraph_" + form + ".adata"), "rb"))

    dat = graph.datasets[nthGen]

    rs.append(dat["simR"])
    vs.append(dat["simV"])

    r, l = compress(dat["elemR"], dat["elemL"], int(DPI * 3.25), 5e-7, max(rs[0]))
    # l0 = dat['elemL']
    # elR = dat['elemR']
    elR = np.array(r[1::3])
    l0 = np.array(l[1::3])

    dnew = {"v" + title: dat["simV"], "r" + title: dat["simR"]}
    dat.update(dnew)

# dat=newGraph.datasets[nthGen]
# rs=[k for k in dat.keys() if k[0]=='r']
# vs=[k for k in dat.keys() if k[0]=='v']
# ls=[k[1:] for k in vs]


# sim=study.load_simulation('sim11')
# l0 = graph.datasets[11]['elemL']
# elR = graph.datasets[11]['elemR']


# %%
with plt.rc_context(
    {
        "lines.markersize": 0.5,
        "lines.linewidth": 1,
        "figure.figsize": figdims,
        "axes.grid": False,
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.prop_cycle": plt.cycler("color", plt.colormaps["tab10"].colors[:4])
        + plt.cycler("linestyle", ["-", "--", ":", "-."])
        # 'axes.prop_cycle': plt.rcParams['axes.prop_cycle'][:4] + plt.cycler('linestyle', ['-', '--', ':', '-.'])
    }
):
    f, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [5, 1]})

    axes[0].set_xscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlim(5e-7, max(rs[0]))

    axes[0].set_ylabel("Absolute error [V]")
    axes[1].set_xlabel("Distance from source")
    axes[1].set_ylabel(r"$\ell_0$ [m]")

    axes[1].scatter(elR, l0, c=xc.colors.BASE, marker=".")

    # xc.visualizers.engineering_ticks(axes[0], 'm', 'V')
    xc.visualizers.engineering_ticks(axes[1], "m", None)

    for r, v, l in zip(rs, vs, titles):
        an = 1e-6 / r
        an[r <= 1e-6] = 1.0
        er = v - an

        rcomp, ercomp = compress(r, er, npx, 5e-7, max(rs[0]))

        axes[0].plot(rcomp, ercomp, label=l)

    axes[0].legend()
    f.align_ylabels()


# study.save_plot(f, 'FormulationErrorsDetail')
