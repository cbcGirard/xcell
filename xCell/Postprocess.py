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
filterCategories=["Mesh type"]
# filterVals=["adaptive","uniform"]
filterVals=['adaptive']

studyPath=datadir+"dual"
# filterCategories=['Element type']
# filterVals=['adaptive']

xmax=1e-4


bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)


    

# aniGraph=study.animatePlot(xCell.error2d,'err2d')
# aniGraph=study.animatePlot(xCell.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xCell.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xCell.centerSlice,'img_mesh')
# aniImg=study.animatePlot(xCell.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xCell.centerSlice,'img_uniform',['Mesh type'],['uniform'])



fig=plt.figure()
plotters=[xCell.Visualizers.ErrorGraph,
            xCell.Visualizers.SliceSet,
            xCell.Visualizers.CurrentPlot]#,
            # xCell.Visualizers.CurrentPlot]

ptype=['ErrorGraph',
        'SliceSet',
        'CurrentShort']#,
        # 'CurrentLong']


xCell.Visualizers.groupedScatter(study.studyPath+'/log.csv',
                     xcat='Number of elements',
                     ycat='Error',
                     groupcat=filterCategories[0])
nufig=plt.gcf()
study.savePlot(nufig, 'AccuracyCost', '.eps')
study.savePlot(nufig, 'AccuracyCost', '.png')


fstack,fratio=xCell.Visualizers.plotStudyPerformance(study)
study.savePlot(fstack, 'Performance', '.eps')
study.savePlot(fstack, 'Performance', '.png')

study.savePlot(fratio, 'Ratio', '.eps')
study.savePlot(fratio, 'Ratio', '.png')


for fv in filterVals:
    for ii,p in enumerate(plotters):
        plt.clf()
        if ii==3:
            plotr=p(fig,study,fullarrow=True)
        else:
            plotr=p(fig,study)
        
        plotr.getStudyData(filterCategories=filterCategories,
                          filterVals=[fv])
        
        name=ptype[ii]+'_'+fv
        
        ani=plotr.animateStudy(name)
        
        


# plotE=xCell.ErrorGraph(plt.figure(),study,{'showRelativeError':True})
# plotE.getStudyData(filterCategories=filterCategories,
#                     filterVals=filterVals)
# aniE=plotE.animateStudy('RelativeError')

# plotE=xCell.ErrorGraph(plt.figure(),study)
# plotE.getStudyData(filterCategories=filterCategories,
#                     filterVals=filterVals)
# aniE=plotE.animateStudy('AbsError-adm')

# plotS=xCell.SliceSet(plt.figure(),study)
# plotS.getStudyData(filterCategories=filterCategories,
#                     filterVals=filterVals)
# aniS=plotS.animateStudy('Slice')


# plotC=xCell.CurrentPlot(plt.figure(),study)
# plotC.getStudyData(filterCategories=filterCategories,
#                           filterVals=filterVals)
# aniC=plotC.animateStudy('CurrentShort')

