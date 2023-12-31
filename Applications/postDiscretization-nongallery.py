#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:19:43 2022

Post-processing of alternate discretizations (FEM, mesh dual,...)

@author: benoit
"""

import xcell as xc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import Common_nongallery as com
import os
import matplotlib as mpl


class Composite(xc.visualizers.FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        super().__init__(fig, study, prefs)

    def showAnalytic(self, ax=None):
        rAnalytic = 0.5 * np.logspace(-6, -3)
        vAnalytic = 1e-6 / rAnalytic
        vAnalytic[rAnalytic < 1e-6] = 1.0

        if ax is None:
            ax = plt.gca()
        ax.semilogx(rAnalytic, vAnalytic, color=xc.colors.BASE, label="Analytic")
        ax.set_xlim(rAnalytic[0], rAnalytic[-1])

        return ax

    def setup_figure(self, resetBounds=False):
        ax = self.showAnalytic()
        self.axes.append(ax)

        xc.visualizers.engineering_ticks(ax, "m", "V")

        ax.set_ylim(bottom=1e-3, top=1e1)
        ax.set_ylabel("Voltage")
        ax.set_xlabel("Distance")

    def get_artists(self, setnumberber, data=None):
        ax = self.axes[0]

        if data is None:
            data = self.datasets[setnumberber]

        titles = ["Admittance", "Trilinear FEM", "Mesh dual"]

        # if titles is None:
        # titles = ['Admittance', 'FEM', 'Dual']

        artists = []
        for n, title in enumerate(titles):
            artists.extend(
                ax.loglog(data["r" + title], data["v" + title], color="C" + str(n), label=title, alpha=0.75)
            )

        if setnumberber == 0:
            if "keepLegend" in self.prefs:
                ax.legend(loc="upper right")
            else:
                artists.append(ax.legend(loc="upper right"))

        return artists


class ErrorComp(xc.visualizers.FigureAnimator):
    def setup_figure(self, resetBounds=False):
        self.fig, self.axes = plt.subplots(2, 1, sharey=True, sharex=True)
        self.axes[0].set_title("Admittance")
        self.axes[1].set_title("Mesh dual")

        [a.set_xscale("log") for a in self.axes]
        [a.set_xlim(left=5e-7) for a in self.axes]

        self.axes[1].set_xlabel("Distance [m]")

    def get_artists(self, setnumberber, data=None):
        ax = self.axes[0]

        if data is None:
            data = self.datasets[setnumberber]

        titles = ["Admittance", "Mesh dual"]

        # if titles is None:
        # titles = ['Admittance', 'FEM', 'Dual']

        artists = []

        for n, title in enumerate(titles):
            vs = data["v" + title]
            r = data["r" + title]

            rana = r.copy()
            rana[rana < 1e-6] = 1e-6
            vAna = 1e-6 / rana

            sel = r > 1e-6

            d = (vs - vAna)[sel]

            artists.append(self.axes[n].fill_between(r[sel], d, color="r", alpha=0.75))

        return artists


folder = "Quals/formulations"
# folder = 'Quals/fixedDisc'
formulations = ["Admittance", "FEM", "Face"]
titles = ["Admittance", "Trilinear FEM", "Mesh dual"]

# folder = 'Quals/PoC'
# formulations = ['Adaptive', 'Uniform']
# titles = ['Octree', 'Uniform']

study, _ = com.makeSynthStudy(folder)
folder = study.study_path
xc.colors.use_dark_style()


# %%

f = plt.figure(
    # figsize=[6.5*.6, 8/3]
    figsize=[6.4, 4.8],
    dpi=150,
)

newGraph = Composite(f, study, prefs={"keeplegend": True})
ax = newGraph.axes[0]


for form, title in zip(formulations, titles):
    graph = pickle.load(open(os.path.join(folder, "ErrorGraph_" + form + ".adata"), "rb"))

    for ii, dat in enumerate(graph.datasets):
        dnew = {"v" + title: dat["simV"], "r" + title: dat["simR"]}
        if ii >= len(newGraph.datasets):
            newGraph.datasets.append(dnew)
        else:
            newGraph.datasets[ii].update(dnew)

    # graphdata=xc.misc.transpose_dicts(graph.datasets)

    # finalData['v'+title]=graphdata['simV']
    # finalData['r'+title]=graphdata['simR']

newGraph.study.study_path = os.getcwd()
ax.set_ylabel("")
newGraph.animate_study("Composite2")  # ,artists = artists)


# %%
c = np.array(plt.colormaps["tab10"].colors[:4])
c[2] = mpl.colors.to_rgb("#990000")
with plt.rc_context(
    {"figure.figsize": [4, 4], "figure.dpi": 250, "font.size": 10, "axes.prop_cycle": plt.cycler("color", c)}
):
    # g=newGraph.copy()
    g = Composite(plt.figure(), study)
    g.study.study_path = os.getcwd()

    g.datasets = newGraph.datasets
    g.prefs.update({"keepLegend": True})
    g.animate_study("Composite")


# %% paired error graphs

with plt.rc_context(
    {
        "figure.figsize": [4, 6],
        "figure.dpi": 250,
        "font.size": 10,
    }
):
    # g=newGraph.copy()
    g = ErrorComp(None, study)
    g.datasets = newGraph.datasets
    g.axes[0].set_xlim(right=max(g.datasets[0]["rAdmittance"]))
    g.animate_study("ErrorAreas")


# %%
xc.colors.use_light_style()
frames = [2, 6, 10]

f, axes = plt.subplots(3, 1, figsize=[6.5, 8], sharex=True)
liteGraph = newGraph.copy(new_figure=f)


for ax, frame in zip(axes, frames):
    liteGraph.axes = [ax]
    liteGraph.showAnalytic(ax)

    # liteGraph.get_artists(frame)
    for t in titles:
        ax.loglog(liteGraph.datasets[frame]["r" + t], liteGraph.datasets[frame]["v" + t], label=t)

    ax.set_ylabel("Potential [V]")
    ax.set_ylim(bottom=1e-3)

    if frame == frames[0]:
        ax.legend()

    if frame == frames[-1]:
        ax.set_xlabel("Distance [m]")


study.save_plot(f, "frames-lite")


# %% Summary graphs
xc.colors.use_light_style()

logfile = study.study_path + "/log.csv"
df, cats = study.load_logfile()

# xaxes=['Number of elements','Total time [Wall]']
# xaxes = ['Number of elements']
# group = 'Element type'
# xaxes=['adaptive','FEM','Face']
# l0string = r'Smallest $\ell_0$ [m]'

group = "Mesh type"
xaxes = ["adaptive", "uniform"]


def logfloor(val):
    return 10 ** np.floor(np.log10(val))


def logceil(val):
    return 10 ** np.ceil(np.log10(val))


def loground(axis, which="both"):
    # xl=axis.get_xlim()
    # yl=axis.get_ylim()

    lims = [[logfloor(aa[0]), logceil(aa[1])] for aa in axis.dataLim.get_points().transpose()]  # [xl, yl]]
    if which == "x" or which == "both":
        axis.set_xlim(lims[0])
    if which == "y" or which == "both":
        axis.set_ylim(lims[1])


# plt.rcParams['axes.prop_cycle'] +cycler('linestyle', ['-', '--', ':', '-.'])

with mpl.rc_context(
    {
        "lines.markersize": 6,
        "lines.linewidth": 2,
        # 'figure.figsize': [3.25, 2],
        "figure.figsize": [6.5, 4.0],
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.prop_cycle": plt.rcParams["axes.prop_cycle"][:4] + plt.cycler("linestyle", ["-", "--", ":", "-."]),
    }
):
    # f, axes = plt.subplots(2, 1, sharey=True)
    f, axes = plt.subplots()

    xc.visualizers.grouped_scatter(
        logfile, x_category="Number of elements", y_category="Error", group_category=group, ax=axes
    )
    # ax.set_ylim(bottom=logfloor(ax.get_ylim()[0]))

    # loground(axes)
    axes.set_yscale("linear")
    loground(axes, which="x")

    # axes.invert_xaxis()
    axes.legend(labels=titles)
    # axes.set_xlabel(l0string)

# study.save_plot(f, 'Error_composite')
study.save_plot(f, "Error_composite-fullpage")

# study.save_plot(f2, 'PerformanceSummary')


# %% Computation time vs element size

# xaxes=['Number of elements','min_l0']
xaxes = ["Number of elements", "Error"]


isoct = df["Mesh type"] == "adaptive"
gentimes = df["Make elements [Wall]"][isoct].to_numpy()
newtime = gentimes - np.roll(gentimes, 1)
newtime[:2] = 0


hatches = ["-", "|", ".", "/", "*", "\\", "||", "--"]
# with mpl.rc_context({'figure.figsize': [6.5, 3]}):
with mpl.rc_context(
    {
        "lines.markersize": 2.5,
        "lines.linewidth": 1,
        "figure.figsize": [3.25, 2.5],
        "font.size": 9,
        "legend.fontsize": 8,
        "axes.prop_cycle": plt.rcParams["axes.prop_cycle"][:8] + plt.cycler("hatch", hatches),
    }
):
    f, axes = plt.subplots(1, 3, sharex=True, sharey=True)

    for ax, mtype, title in zip(axes, ["Admittance", "FEM", "Face"], titles):
        ax.set_title(title, fontsize=10)
        xc.visualizers.import_and_plot_times(
            fname=logfile,
            time_type="Wall",
            ax=ax,
            x_category="Number of elements",
            only_category="Element type",
            only_value=mtype,
            hatch=hatches,
        )

[a.set_ylabel("") for a in axes[1:]]
axes[0].set_xlim(left=0)
[a.set_xlabel("Number of\nelements") for a in axes]


xc.visualizers.outsideLegend(axes[1], reverse_order=False, where="bottom", fontsize=8, ncol=2)

f.subplots_adjust(bottom=0.525, wspace=0.2)
newLeg = [
    "Make elements",
    "Finalize mesh",
    "Calculate conductances",
    "Renumber nodes",
    "Sort node types",
    "Filter conductances",
    "Assemble system",
    "Solve system",
]
[t.set_text(l) for t, l in zip(axes[1].legend_.get_texts(), newLeg)]

study.save_plot(f, "discretPerformanceStack")
