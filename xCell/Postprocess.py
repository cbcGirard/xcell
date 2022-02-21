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

studyPath=datadir+"dualComp"
filterCategories=['Element type']
filterVals=['Admittance','Face']

xmax=1e-4


bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)




# aniGraph=study.animatePlot(xCell.error2d,'err2d')
# aniGraph=study.animatePlot(xCell.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xCell.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xCell.VisualizerscenterSlice,'img_mesh')
# aniImg=study.animatePlot(xCell.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xCell.centerSlice,'img_uniform',['Mesh type'],['uniform'])




# plotters=[xCell.Visualizers.ErrorGraph,
#             xCell.Visualizers.SliceSet,
#             xCell.Visualizers.CurrentPlot]#,
#             # xCell.Visualizers.CurrentPlot]

# ptype=['ErrorGraph',
#         'SliceSet',
#         'CurrentShort']#,
#         # 'CurrentLong']

plotters=[xCell.Visualizers.ErrorGraph]

xCell.Visualizers.groupedScatter(study.studyPath+'/log.csv',
                     xcat='Number of elements',
                     ycat='Error',
                     groupcat=filterCategories[0])
nufig=plt.gcf()
study.savePlot(nufig, 'AccuracyCost', '.eps')
study.savePlot(nufig, 'AccuracyCost', '.png')


# fstack,fratio=xCell.Visualizers.plotStudyPerformance(study)
# study.savePlot(fstack, 'Performance', '.eps')
# study.savePlot(fstack, 'Performance', '.png')

# study.savePlot(fratio, 'Ratio', '.eps')
# study.savePlot(fratio, 'Ratio', '.png')


for fv in filterVals:
    
    fstack,fratio=xCell.Visualizers.plotStudyPerformance(study,
                                                         onlyCat=filterCategories[0],
                                                         onlyVal=fv)
    fstem='_'+filterCategories[0]+str(fv)
    
    study.savePlot(fstack, 'Performance'+fstem, '.eps')
    study.savePlot(fstack, 'Performance'+fstem, '.png')

    study.savePlot(fratio, 'Ratio'+fstem, '.eps')
    study.savePlot(fratio, 'Ratio'+fstem, '.png')
    
    

        
        
fig=plt.figure()

for ii,p in enumerate(plotters):
    plt.clf()        
    plotr=p(fig,study)
    artistSets=[]
    fileNames=[]

    for fv in filterVals:
        plotr.getStudyData(filterCategories=filterCategories,
                          filterVals=[fv])
        
        artistSets.append(plotr.getArtists())
        fileNames.append(p.__name__+'_'+str(fv))
        
    for name,art in zip(fileNames,artistSets):
        ani=plotr.animateStudy(fname=name,artists=art)
        
        # ani=plotr.animateStudy(name)