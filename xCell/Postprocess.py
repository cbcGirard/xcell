#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:24:52 2022

@author: benoit
"""


import numpy as np
import numba as nb
import xCell
import matplotlib.pyplot as plt

meshtype='adaptive'
datadir='/home/benoit/smb4k/ResearchData/Results/studyTst/'
# studyPath=datadir+'regularization/'
# filterCategories=["Regularized?"]
# filterVals=[True]

# studyPath=datadir+'admVsFEM-Voltage'
# studyPath=datadir+'femVadmit'
# filterCategories=['Element type']
# filterVals=['Admittance', 'FEM']

# studyPath=datadir+'vsrc'
# filterCategories=['Source']
# filterVals=['current','voltage']

# studyPath=datadir+'uniVsAdapt'
# filterCategories=["Mesh type"]
# filterVals=["adaptive","uniform"]
# filterVals=['adaptive']

# studyPath=datadir+"dualComp"
# filterCategories=['Element type']
# filterVals=['Face','Admittance']

# studyPath=datadir+'NEURON'
# filterCategories=None
# filterVals=None

studyPath=datadir+'post-renumber'
filterCategories=None
filterVals=[None]

xmax=1e-4


bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)




# aniGraph=study.animatePlot(xCell.error2d,'err2d')
# aniGraph=study.animatePlot(xCell.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xCell.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xCell.VisualizerscenterSlice,'img_mesh')
# aniImg=study.animatePlot(xCell.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xCell.centerSlice,'img_uniform',['Mesh type'],['uniform'])


# staticPlots=True
staticPlots=False

plotters=[xCell.Visualizers.ErrorGraph,
            xCell.Visualizers.SliceSet,
            xCell.Visualizers.CurrentPlot]


# plotters=[xCell.Visualizers.ErrorGraph]

# ptr=xCell.Visualizers.ErrorGraph(plt.figure(), study)

# # ptr=xCell.Visualizers.SliceSet(plt.figure(), study)
# ptr.getStudyData(sortCategory=filterCategories[0])
# ani=ptr.animateStudy()


if staticPlots:    
    xCell.Visualizers.groupedScatter(study.studyPath+'/log.csv',
                         xcat='Number of elements',
                         ycat='Error',
                         groupcat=filterCategories[0])
    nufig=plt.gcf()
    study.savePlot(nufig, 'AccuracyCost', '.eps')
    study.savePlot(nufig, 'AccuracyCost', '.png')
    
    for fv in filterVals:
        
        fstack,fratio=xCell.Visualizers.plotStudyPerformance(study,
                                                             onlyCat=filterCategories[0],
                                                             onlyVal=fv)
        fstem='_'+filterCategories[0]+str(fv)
        
        study.savePlot(fstack, 'Performance'+fstem, '.eps')
        study.savePlot(fstack, 'Performance'+fstem, '.png')
    
        study.savePlot(fratio, 'Ratio'+fstem, '.eps')
        study.savePlot(fratio, 'Ratio'+fstem, '.png')
        
        

        
        

# #THIS WORKS
# for ii,p in enumerate(plotters):

#     for fv in filterVals:
#         fname=p.__name__+'_'+str(fv)
        
        
#         plotr=p(plt.figure(),study)


#         plotr.getStudyData(filterCategories=filterCategories,
#                   filterVals=[fv])
#         plotr.animateStudy(fname=fname)

        

for ii,p in enumerate(plotters):
    
    plots=[]
    names=[]
    ranges=None
    for fv in filterVals:
        fname=p.__name__+'_'+str(fv)
        plotr=p(plt.figure(),study)

        plotr.getStudyData(filterCategories=filterCategories,
                  filterVals=[fv])
        
        plots.append(plotr)
        names.append(fname)
        
        if ranges is not None:
            plotr.unifyScales(ranges)
        ranges=plotr.dataScales
        
    for plot,name in zip(plots,names):
        plot.dataScales=ranges
        
        plot.animateStudy(fname=name)