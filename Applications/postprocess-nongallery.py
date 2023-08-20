#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:24:52 2022

@author: benoit
"""


import numpy as np
import numba as nb
import xcell
import matplotlib.pyplot as plt

meshtype = "adaptive"
datadir = "/home/benoit/smb4k/ResearchData/Results/studyTst/"
# study_path=datadir+'regularization/'
# filter_categories=["Regularized?"]
# filter_values=[True]

# study_path=datadir+'admVsFEM-Voltage'
# study_path=datadir+'femVadmit'
# filter_categories=['Element type']
# filter_values=['Admittance', 'FEM']

# study_path=datadir+'vsrc'
# filter_categories=['Source']
# filter_values=['current','voltage']

# study_path = datadir+'uniVsAdapt'
study_path = "/home/benoit/smb4k/ResearchData/Results/Quals/PoC"
filter_categories = ["Mesh type"]
filter_values = ["adaptive", "uniform"]
# filter_values=['adaptive']

# study_path=datadir+"dualComp2"
# filter_categories=['Element type']
# filter_values=['Face','Admittance']
# filter_values=['Admittance']

# study_path=datadir+'NEURON'
# filter_categories=None
# filter_values=None

# study_path=datadir+'post-renumber'
# filter_categories=None
# filter_values=[None]

# study_path=datadir+'quickie'
# filter_categories=None
# filter_values=[None]

# study_path = datadir+'Boundary_large/rubik0'
# filter_categories = ['Boundary']
# # filter_values=['Analytic','Ground','Rubik0']
# filter_values = ['Analytic']  # ,'Ground','Rubik0']


xmax = 1e-4


bbox = np.append(-xmax * np.ones(3), xmax * np.ones(3))


study = xcell.Study(study_path, bbox)


# aniGraph=study.animatePlot(xcell.error2d,'err2d')
# aniGraph=study.animatePlot(xcell.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xcell.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xcell.VisualizerscenterSlice,'img_mesh')
# aniImg=study.animatePlot(xcell.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xcell.centerSlice,'img_uniform',['Mesh type'],['uniform'])


# staticPlots = True
staticPlots = False

plotters = [
    xcell.visualizers.ErrorGraph,
    #     xcell.Visualizers.SliceSet,
    # xcell.Visualizers.CurrentPlot,
    # xcell.Visualizers.LogError
]


# plotters=[xcell.Visualizers.ErrorGraph]

# ptr=xcell.Visualizers.ErrorGraph(plt.figure(), study)

# # ptr=xcell.Visualizers.SliceSet(plt.figure(), study)
# ptr.get_study_data(sort_category=filter_categories[0])
# ani=ptr.animate_study()

# %%


if staticPlots:
    xcell.Visualizers.grouped_scatter(
        study.study_path + "/log.csv",
        x_category="Number of elements",
        y_category="FVU",
        group_category=filter_categories[0],
    )
    nufig = plt.gcf()
    study.save_plot(nufig, "AccuracyCost", ".eps")
    study.save_plot(nufig, "AccuracyCost", ".png")

    for fv in filter_values:
        fstack, fratio = xcell.visualizers.plot_study_performance(
            study, only_category=filter_categories[0], only_value=fv
        )
        fstem = "_" + filter_categories[0] + str(fv)

        study.save_plot(fstack, "Performance" + fstem, ".eps")
        study.save_plot(fstack, "Performance" + fstem, ".png")

        study.save_plot(fratio, "Ratio" + fstem, ".eps")
        study.save_plot(fratio, "Ratio" + fstem, ".png")


# #THIS WORKS
# for ii,p in enumerate(plotters):

#     for fv in filter_values:
#         fname=p.__name__+'_'+str(fv)


#         plotr=p(plt.figure(),study)


#         plotr.get_study_data(filter_categories=filter_categories,
#                   filter_values=[fv])
#         plotr.animate_study(fname=fname)


for ii, p in enumerate(plotters):
    plots = []
    names = []
    ranges = None
    for fv in filter_values:
        fname = p.__name__ + "_" + str(fv)
        plotr = p(plt.figure(), study)
        if "universalPts" in plotr.prefs:
            plotr.prefs["universalPts"] = True

        plotr.get_study_data(filter_categories=filter_categories, filter_values=[fv])

        plots.append(plotr)
        names.append(fname)

        if ranges is not None:
            plotr.unify_scales(ranges)
        ranges = plotr.data_scales

    for plot, name in zip(plots, names):
        plot.data_scales = ranges

        plot.animate_study(fname=name + "redux")
