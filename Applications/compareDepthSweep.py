#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:46:30 2022

@author: benoit
"""

import xcell
import Common
import matplotlib.pyplot as plt
import argparse
import numpy as np

cli=argparse.ArgumentParser()
cli.add_argument('--comparison', choices=['bounds','mesh','formula'], default='testing')
cli.add_argument('-p','--plot-only',help='skip simulation and use existing data', action = 'store_true')
# cli.add_argument('-a','--animate',help='skip simulation and use existing data', action = 'store_true')
# cli.add_argument('-p','--plot-only',help='skip simulation and use existing data', action = 'store_true')

args = cli.parse_args()


generate = True

# plot performance info
staticPlots = True

depths=np.arange(3,18)

if args.comparison=='mesh':
    foldername = 'Quals/PoC'
    tstVals=["adaptive","uniform"]
    # tstVals=['adaptive','equal elements',r'equal $l_0$']
    tstCat='Mesh type'
if args.comparison=='formula':
    foldername = 'Quals/formulations'
    tstVals = ['Admittance', 'FEM', 'Face']
    tstCat = 'Element type'
if args.comparison == 'bounds':
    foldername = 'Quals/boundaries'
    tstVals=['Analytic','Ground','Rubik0']
    tstCat='Boundary'
if args.comparison == 'testing':
    foldername = 'Quals/miniset'
    tstVals = ['adaptive', 'uniform']
    tstCat = 'Mesh type'
    generate = False
    staticPlots = False
    depths=np.arange(3,8)


# if args.comparison=='voltage':
    # tstVals=[False, True]
    # tstCat='Vsrc?'


# generate animation(s)
plotters = [
    xcell.visualizers.ErrorGraph,
    # xcell.visualizers.ErrorGraph,
    # xcell.visualizers.SliceSet,
    # xcell.visualizers.LogError,
    # xcell.visualizers.CurrentPlot,
]

plotPrefs = [
    None,
    # {'onlyDoF':True},
    # None,
    # None,
]




study, _ = Common.makeSynthStudy(foldername)
#%%

if generate and not args.plot_only:
    Common.pairedDepthSweep(study,
                            depthRange=depths,
                            testCat=tstCat,
                            testVals=tstVals)

#%%

costcat = 'Error'
# costcat='FVU'
# xcat='l0min'

xcell.colors.useLightStyle()

xvalues=['Number of elements','l0min','Total time [Wall]']
xtags=['numel','l0','totTime']
if staticPlots:
    for xcat,xtag in zip(xvalues,xtags):


        xcell.visualizers.groupedScatter(study.studyPath+'/log.csv',
                                         xcat=xcat,
                                         ycat=costcat,
                                         groupcat=tstCat)
        fname = tstCat+"_"+costcat+'-vs-'+xtag
        fname.replace(' ', '_')
        nufig = plt.gcf()
        study.savePlot(nufig, fname)
        for fv in tstVals:


            fstack, fratio = xcell.visualizers.plotStudyPerformance(study,
                                                                    onlyCat=tstCat,
                                                                    onlyVal=fv)
            fstem = '_'+tstCat+str(fv)

            study.savePlot(fstack, 'Performance'+fstem)

            study.savePlot(fratio, 'Ratio'+fstem)




#%%
xcell.colors.useDarkStyle()


for ii, p in enumerate(plotters):

    plots = []
    names = []
    ranges = None
    for fv in tstVals:
        fname = p.__name__+'_'+str(fv)
        fname.replace(' ', '_')
        plotr = p(plt.figure(), study, prefs = plotPrefs[ii])
        if 'universalPts' in plotr.prefs:
            plotr.prefs['universalPts'] = True
        if 'onlyDoF' in plotr.prefs:
            if plotr.prefs['onlyDoF']:
                fname+='-detail'


        plotr.getStudyData(filterCategories=[tstCat],
                           filterVals=[fv])

        plots.append(plotr)
        names.append(fname)

        if ranges is not None:
            plotr.unifyScales(ranges)
        ranges = plotr.dataScales

    for plot, name in zip(plots, names):
        plot.dataScales = ranges

        xcell.colors.useDarkStyle()
        plot.animateStudy(fname=name, fps=1.0)

        xcell.colors.useLightStyle()
        nuprefs={}
        if ii<2:
            nuprefs.update({'printScale':[2,3],})

        liteplot=plot.copy(nuprefs)
        liteplot.animateStudy(fname=name+'-lite', fps=1)