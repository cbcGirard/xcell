#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:06:06 2022

Postprocessing of dynamic adaptation tests

@author: benoit
"""

import xcell
import Common_nongallery
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import xcell.nrnutil as nUtil

study_path = "Quals/polyCell"
nRing = 5
ring = Common_nongallery.Ring(N=nRing, stim_delay=0, dendSegs=101, r=175)
tstop = 40
tPerFrame = 5
barRatio = [9, 1]

fullwidth = True
if fullwidth:
    widthtag = "-fullwidth"
else:
    widthtag = ""

skipExt = False


# study_path = 'Quals/monoCell'
# nRing = 0
# tstop = 12
# tPerFrame = 2
# barRatio = [4, 1]

study, _ = Common_nongallery.makeSynthStudy(study_path)
folderstem = study.study_path


dmin = 4
dmin = 8


comDict = study.load("commonData", ".xstudy")
tv = comDict["tv"]
I = comDict["I"]
rads = comDict["rads"]
coords = comDict["coords"]
is_sphere = comDict["is_sphere"]
polygons = comDict["polygons"]
# %%


def getdata(strat):
    fname = os.path.join(folderstem, strat, strat + ".pcr")
    data = pickle.load(open(fname, "rb"))
    return data


labels = [l for l in os.listdir(folderstem) if os.path.isdir(os.path.join(folderstem, l))]
labels.sort()
labels = [labels[n] for n in [3, 0, 2, 1]]
data = [getdata(l) for l in labels]


# categories=[d['depths'] for d in data]
# Override category labels
categories = ["Fixed\n[4]", "Dynamic\n[4:8]", "Fixed\n[8]", "Dynamic\n[4:12]"]

if skipExt:
    data.pop()
    labels.pop()

for lite in ["lite"]:  # ,'-lite']:
    if lite == "":
        xcell.colors.use_dark_style()
    else:
        xcell.colors.use_light_style()

    f, ax = plt.subplots(3, gridspec_kw={"height_ratios": [5, 5, 2]}, layout="constrained")
    ax[2].plot(tv, I)

    for d, l in zip(data, categories):
        ax[0].plot(tv, np.abs(d["intErr"]), label=l.replace("\n", " "))
        ax[1].plot(tv[1:], d["dt"][1:])

    ax[0].legend()
    ax[0].set_ylabel("Error")
    ax[1].set_ylabel("Wall time")
    ax[2].set_ylabel("Activity")
    ax[2].set_xlabel("Simulated time")

    xcell.visualizers.engineering_ticks(ax[2], xunit="s")
    for a in ax[:-1]:
        a.set_xticks([])
        a.grid(True, axis="both")

    # bar plot
    with plt.rc_context({"font.size": 12, "figure.figsize": [6, 4]}):
        errColor = "#990000"
        # errColor='C1'
        f2, ax = plt.subplots(1, 1)
        aright = ax.twinx()

        # [a.set_yscale('log') for a in [ax,aright]]

        ttime = [sum(l["dt"]) for l in data]
        terr = [sum(np.abs(l["intErr"])) for l in data]
        # terr=[sum(np.abs(l['intErr']))/sum(np.abs(l['intAna'])) for l in data]
        # terr=[sum(np.abs(l['SSE']))/sum(np.abs(l['SSTot'])) for l in data]

        barpos = np.arange(len(categories))
        barw = 0.25

        tart = ax.bar(barpos - barw / 2, height=ttime, width=barw, color="C0", label="Time")
        ax.set_ylabel("Total simulation time")

        eart = aright.bar(barpos + barw / 2, height=terr, width=barw, color=errColor, label="Error")
        aright.set_ylabel("Total simulation error")

        ax.set_xticks(barpos)
        ax.set_xticklabels(categories)

        ax.legend(handles=[tart, eart])

        axes = [ax, aright]
        ntick = 0

        nticks = [len(a.get_yticks()) for a in axes]
        ntick = max(nticks)
        for a in axes:
            dtick = a.get_yticks()[1]
            a.set_yticks(np.arange(ntick) * dtick)
            a.grid(axis="x")

        study.save_plot(f2, os.path.join(folderstem, "ring%dsummary%s" % (nRing, lite) + widthtag))

        study.save_plot(f, os.path.join(folderstem, "ring%dcomparison%s" % (nRing, lite) + widthtag))


# #%% Semi- manual relabling
# for l in labels:
#     d=getdata(l)
#     if l[:5]=='fixed':
#         if l[5:]=='Min':
#             dstr=
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

# %% voltage plots
an = pickle.load(open(os.path.join(folderstem, "depth", "volt-depth.adata"), "rb"))
# fname = 'volt-depth'
fname = "tmp"

xcell.colors.use_light_style()

# alite = an.copy({'colorbar': False,
#                  'barRatio': barRatio})

alite = an
# %%
f = plt.figure(figsize=(2.0, 2.0), dpi=300, tight_layout=False)

ax = f.add_axes([0, 0, 1, 1], xticks=[], yticks=[])
alite.axes = [ax]
alite.fig = f
vizlim = 4e-4
ax.set_ylim(-vizlim, vizlim)
ax.set_xlim(-vizlim, vizlim)

# figw = 0.3*6.5

# alite.tbar.axes[0].set_xlim(right=tstop/1000)


_ = nUtil.show_cell_geo(alite.axes[0])

# get closest frames to 5ms intervals
frameNs = [int(f * len(alite.datasets) / tstop) for f in np.arange(0, tstop, tPerFrame)]
frameNs.append(len(alite.datasets) - 1)
# artists = [alite.get_artists(ii) for ii in frameNs]
# alite.animate_study(fname+'-lite', fps=30, artists=artists,
#                     vector_frames=np.arange(len(frameNs)), unitStr='V')

for ii in range(len(frameNs)):
    alite.reset_figure()
    # figw = 2.
    # alite.fig.set_figwidth(figw)
    # alite.fig.set_figheight(1.*figw)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # alite.tbar.axes[0].set_xlim(right=tstop/1000)
    alite.get_artists(frameNs[ii])
    nUtil.show_cell_geo(ax)

    tag = ax.text(
        0.1,
        0.9,
        "%d ms" % (5 * ii),
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round"),
    )

    alite.fig.canvas.draw()
    alite.fig.savefig("f%d.eps" % ii, dpi=300)
    tag.remove()

# %% voltage for screen

# xcell.colors.CM_BIPOLAR=xcell.colors.scoop_cmap(cm.guppy_r,0.5)

viz = pickle.load(open(folderstem + "/depth/volt-depth.adata", "rb"))


with plt.rc_context(
    {
        "figure.dpi": 144,
        # 'figure.figsize':[6.75,7.5],
        "figure.figsize": [6.75, 5.0],
        "toolbar": "none",
    }
):
    barRatio = [4, 1]
    vd = viz.copy({"colorbar": False, "barRatio": barRatio, "labelAxes": False})
    # vd.data_scales['spaceV'].knee=1e-8
    # vd.data_scales['spaceV'].min=-1e-5
    ax = vd.axes[0]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    nUtil.show_cell_geo(ax)

    vd.get_artists(0)

    vd.animate_study("Showcase")
