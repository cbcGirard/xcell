#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:46:30 2022

@author: benoit
"""

import xcell
import Common
import matplotlib.pyplot as plt

foldername = 'Quals/formulations'

generate = True

# plot performance info
staticPlots = True

# generate animation(s)
plotters = [
    xcell.visualizers.LogError,
    xcell.visualizers.ErrorGraph,
    xcell.visualizers.SliceSet,
    # xcell.visualizers.CurrentPlot,
]

plotPrefs = [
    None,
    # {'onlyDoF':True, 'colorNodeConnectivity':True},
    None,
    None,
    None
]

# tstVals=['adaptive']
# tstVals=["adaptive","uniform"]
# tstVals=['adaptive','equal elements',r'equal $l_0$']
# tstCat='Mesh type'


# tstVals=[False, True]
# tstCat='Vsrc?'

tstVals = ['Admittance', 'FEM', 'Face']
# tstVals=['Admittance','Face']
# # tstVals=['Admittance']
tstCat = 'Element type'

# tstVals=[None]
# tstCat='Power'

# tstVals=['Analytic','Ground']
# tstVals=['Analytic','Ground','Rubik0']
# tstCat='Boundary'


study, _ = Common.makeSynthStudy(foldername)

if generate:
    Common.pairedDepthSweep(study,
                            depthRange=range(3, 18),
                            testCat=tstCat,
                            testVals=tstVals)


costcat = 'Error'
# costcat='FVU'
# xcat='l0min'
xcat = 'Number of elements'
filterCategories = [tstCat]
if staticPlots:
    xcell.visualizers.groupedScatter(study.studyPath+'/log.csv',
                                     xcat=xcat,
                                     ycat=costcat,
                                     groupcat=tstCat)
    fname = tstCat+"_"+costcat+'-vs-'+xcat
    nufig = plt.gcf()
    study.savePlot(nufig, fname, '.eps')
    study.savePlot(nufig, fname, '.png')

    for fv in tstVals:

        fstack, fratio = xcell.visualizers.plotStudyPerformance(study,
                                                                onlyCat=filterCategories[0],
                                                                onlyVal=fv)
        fstem = '_'+filterCategories[0]+str(fv)

        study.savePlot(fstack, 'Performance'+fstem, '.eps')
        study.savePlot(fstack, 'Performance'+fstem, '.png')

        study.savePlot(fratio, 'Ratio'+fstem, '.eps')
        study.savePlot(fratio, 'Ratio'+fstem, '.png')


# #THIS WORKS
# for ii,p in enumerate(plotters):

#     for fv in tstVals:
#         fname=p.__name__+'_'+str(fv)


#         plotr=p(plt.figure(),study)


#         plotr.getStudyData(filterCategories=filterCategories,
#                   tstVals=[fv])
#         plotr.animateStudy(fname=fname)


for ii, p in enumerate(plotters):

    plots = []
    names = []
    ranges = None
    for fv in tstVals:
        fname = p.__name__+'_'+str(fv)
        plotr = p(plt.figure(), study)
        if 'universalPts' in plotr.prefs:
            plotr.prefs['universalPts'] = True

        plotr.getStudyData(filterCategories=filterCategories,
                           filterVals=[fv])

        plots.append(plotr)
        names.append(fname)

        if ranges is not None:
            plotr.unifyScales(ranges)
        ranges = plotr.dataScales

    for plot, name in zip(plots, names):
        plot.dataScales = ranges

        plot.animateStudy(fname=name)
