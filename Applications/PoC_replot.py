#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:45:31 2022

Better plotting of the stacked error plots for print

@author: benoit
"""

import xcell as xc
import Common_nongallery as com
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


study, _ = com.makeSynthStudy("Quals/PoC/")
# study, _ = com.makeSynthStudy('Quals/formulations/')

# study, _ = com.makeSynthStudy('Quals/bigPOC/')

titles = [
    "Octree",
    "Uniform",
]

vizU = study.load("ErrorGraph_uniform", ".adata")
vizA = study.load("ErrorGraph_adaptive", ".adata")


# %%
frames = [2, 6, 14]
compress = False

analytic = vizA.analytic
analyticR = vizA.analyticR

xc.colors.use_light_style()

with plt.rc_context(
    {
        "font.size": 9,
        "axes.grid": False,
        # 'figure.dpi': 200,
        "figure.subplot.hspace": 0.01,
        "toolbar": "None",
    }
):
    f = plt.figure(
        figsize=[6.5, 8.5],
        # tight_layout=True,
        constrained_layout=True,
    )
    hpx = int(3.25 * f.dpi)
    pxVec = np.geomspace(analyticR[0], analyticR[-1], hpx)
    # subs=f.subfigures(3,2)#,wspace=0.25,hspace=0.125)

    # for nrow,row in enumerate(subs):
    #     for v,sub in zip([vizU,vizA],row):

    #         lastRow=nrow==2

    #         v.fig=sub
    #         v.setup_figure(labelX=lastRow, labelY=v==vizU)
    #         v.prefs['showLegend']=(v==vizA and nrow==0)
    #         v.analytic=analytic
    #         v.analyticR=analyticR

    #         v.get_artists(frames[nrow])

    ratios = [5, 5, 3, 0]
    hratio = ratios * 3
    hratio.pop()

    subs = f.subplots(len(hratio), 2, gridspec_kw={"height_ratios": hratio})

    l0span = xc.visualizers.ScaleRange()

    def scootSubplot(axis, scootUp=True, fraction=0.05):
        # pos=axis.get_position().extents
        # #left, bottom, right, top
        # if scootUp:
        #     shift=np.array([0,-1.,0,0])
        # else:
        #     shift=np.array([0,0,0,1.])

        pos = axis.get_position().bounds
        if scootUp:
            shift = np.array([0, -1, 0, -1])
        else:
            shift = np.array([0, 1, 0, -1])

        axis.set_position(pos + fraction * shift)

    for nrow in range(3):
        lastRow = nrow == 2
        for ncol in range(2):
            if ncol:
                v = vizA
                tstr = "Octree"
            else:
                v = vizU
                tstr = "Uniform"

            if nrow == 0:
                subs[nrow, ncol].set_title(tstr)

            v.axes = subs[4 * nrow : 4 * nrow + 3, ncol]
            v.setup_figure(
                labelX=lastRow,
                labelY=v == vizU,
                # labelY=False,
                newAxes=False,
            )

            v.prefs["showLegend"] = False
            v.analytic = analytic
            v.analyticR = analyticR

            data = v.datasets[frames[nrow]]

            if compress:
                data["simV"] = np.interp(pxVec, data["simR"], data["simV"])
                data["simR"] = pxVec.copy()

                maxerr = []
                pxList = []

                l0 = []
                l0r = []
                for ii in range(pxVec.shape[0]):
                    if ii:
                        p0 = pxVec[ii - 1]
                    else:
                        p0 = 0
                    isin = np.logical_and(data["errR"] >= p0, data["errR"] <= pxVec[ii])
                    hasEl = np.logical_and(data["elemR"] >= p0, data["elemR"] <= pxVec[ii])
                    if xc.util.fastcount(isin):
                        whr = np.argmax(np.abs(data["errors"][isin]))
                        maxerr.append(data["errors"][isin][whr])
                        pxList.append(pxVec[ii])
                    if xc.util.fastcount(hasEl):
                        l0.append(np.mean(data["elemL"][hasEl]))
                        l0r.append(np.mean(data["elemR"][hasEl]))

                data["errors"] = np.array(maxerr)
                data["errR"] = np.array(pxList)
                data["elemL"] = np.array(l0)
                data["elemR"] = np.array(l0r)

            v.get_artists(frames[nrow])

            l0span.update(data["elemL"])

    subs[0, 1].legend()
    [ax.set_ylim(l0span.min, l0span.max) for row in subs[2::4] for ax in row]
    [ax.axis("off") for row in subs[3::4] for ax in row]
    [ax.grid(False) for row in subs for ax in row]
    # for ax in subs.ravel():
    #     for t in ax.get_yticklabels():
    #         t.set_rotation(45)
    # # for ii, row in enumerate(subs):
    #     if ii%3==1:
    #         continue
    #     for sub in row:
    #         scootSubplot(sub,ii%3)

    f.align_labels()


# %% Error vs. numel and time
xc.colors.use_light_style()

logfile = study.study_path + "/log.csv"
df, cats = study.load_logfile()

# xaxes=['Number of elements','Total time [Wall]']
xaxes = ["Number of elements", "min_l0"]
group = "Mesh type"

# group='Element type'
# xaxes=['adaptive','FEM','Face']
l0string = r"Smallest $\ell_0$ [m]"


def logfloor(val):
    return 10 ** np.floor(np.log10(val))


def logceil(val):
    return 10 ** np.ceil(np.log10(val))


def loground(axis):
    # xl=axis.get_xlim()
    # yl=axis.get_ylim()

    lims = [[logfloor(aa[0]), logceil(aa[1])] for aa in axis.dataLim.get_points().transpose()]  # [xl, yl]]
    axis.set_xlim(lims[0])
    axis.set_ylim(lims[1])


with mpl.rc_context({"lines.markersize": 5, "lines.linewidth": 2}):
    f, axes = plt.subplots(2, 1, sharey=True, figsize=[6.5, 8])
    f2, a2 = plt.subplots(1, 1, figsize=[6.5, 3])

    for ax, x_category in zip(axes, xaxes):
        xc.visualizers.grouped_scatter(logfile, x_category, y_category="Error", group_category=group, ax=ax)
        # ax.set_ylim(bottom=logfloor(ax.get_ylim()[0]))

    xc.visualizers.grouped_scatter(
        logfile, y_category="Total time [Wall]", x_category="min_l0", group_category=group, ax=a2, df=df[2:]
    )

[loground(a) for a in axes]


axes[1].invert_xaxis()
axes[0].legend(labels=titles)
axes[1].set_xlabel(l0string)
a2.legend(loc="lower right", labels=titles)
loground(a2)
a2.set_xlabel(l0string)

a2.invert_xaxis()

study.save_plot(f, "Error_composite")
study.save_plot(f2, "PerformanceSummary")

# %% Computation time vs element size
# xaxes=['Number of elements','Error']
xaxes = ["Number of elements", "min_l0"]


isoct = df["Mesh type"] == "adaptive"
gentimes = df["Make elements [Wall]"][isoct].to_numpy()
newtime = gentimes - np.roll(gentimes, 1)
newtime[:2] = 0

with mpl.rc_context({"figure.figsize": [6.5, 3]}):
    f, axes = plt.subplots(1, 2, sharex=True)
    # f.subplots_adjust(bottom=0.3)

    # f, axes = plt.subplots(2, 1,sharex=True)#, gridspec_kw={'height_ratios':[2,4]})
    # axes[1,0].invert_xaxis()

    # for arow,x_category in zip(axes,xaxes):

    #     arow[1].sharex(arow[0])
    #     arow[1].sharey(arow[0])
    #     for ax, mtype, title in zip(arow, ['uniform', 'adaptive'], ['Uniform', 'Octree']):
    for ax, mtype, title in zip(axes, ["adaptive", "uniform"], titles):
        # if x_category==xaxes[1]:
        #     xc.visualizers.grouped_scatter(fname=logfile, x_category=x_category, y_category='Total time [Wall]', group_category='Mesh type', ax=ax)
        #     ax.set_yscale('log')
        #     ax.set_xscale('log')
        #     # pass
        # else:
        ax.set_title(title)
        xc.visualizers.import_and_plot_times(
            fname=logfile,
            time_type="Wall",
            ax=ax,
            x_category="Number of elements",
            only_category="Mesh type",
            only_value=mtype,
        )

axes[1].set_ylabel("")
f.align_labels()
axes[0].set_xlim(left=0)


# axes[1].stackplot(df['Number of elements'][isoct], newtime, baseline='zero',color=0.5*np.ones(4))
# xc.visualizers.outsideLegend(where='bottom',ncol=3)
# plt.tight_layout()

study.save_plot(f, "PerformanceStack")

# %% Computation time vs element size plus parallelism
# xaxes=['Number of elements','Error']
xaxes = ["Number of elements", "min_l0"]


isoct = df["Mesh type"] == "adaptive"
gentimes = df["Make elements [Wall]"][isoct].to_numpy()
newtime = gentimes - np.roll(gentimes, 1)
newtime[:2] = 0

with mpl.rc_context({"figure.figsize": [6.5, 5]}):
    # , gridspec_kw={'height_ratios':[2,4]})
    f, axes = plt.subplots(2, 2, sharex=True)
    # axes[1,0].invert_xaxis()

    for arow, x_category in zip(axes, xaxes):
        arow[1].sharey(arow[0])
        for ax, mtype, title in zip(arow, ["uniform", "adaptive"], titles):
            if x_category == xaxes[1]:
                xc.visualizers.import_and_plot_times(
                    fname=logfile,
                    time_type="Ratio",
                    ax=ax,
                    x_category="Number of elements",
                    only_category="Mesh type",
                    only_value=mtype,
                )

                ax.set_yscale("linear")
                # ax.set_xscale('log')
                # pass
            else:
                ax.set_title(title)
                xc.visualizers.import_and_plot_times(
                    fname=logfile,
                    time_type="Wall",
                    ax=ax,
                    x_category="Number of elements",
                    only_category="Mesh type",
                    only_value=mtype,
                )
                ax.set_xlabel("")

            # if mtype='adaptive':
        arow[1].set_ylabel("")


f.align_labels()
axes[0, 0].set_xlim(left=0)

# axes[1].stackplot(df['Number of elements'][isoct], newtime, baseline='zero',color=0.5*np.ones(4))


# %% Image slice redux

with plt.rc_context({"figure.figsize": [7, 6.5], "figure.dpi": 144}):
    # v2=v.copy(override_prefs={'logScale':True})
    study, _ = com.makeSynthStudy("Quals/bigPOC")
    v = xc.visualizers.SliceSet(None, study)
    v.get_study_data()
    v.animate_study("ImageTst")

    # v2.prefs['logScale']=True
    # v2.data_scales['vbounds'].knee=1e-3
    # v2.data_scales['errbounds'].knee=0.1
    # v2.animate_study('Image')
