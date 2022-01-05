#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:56:54 2021

@author: benoit
"""

import numpy as np
import numba as nb
import xCell
import matplotlib.pyplot as plt


meshtype='adaptive'
# studyPath='Results/studyTst/miniCur/'#+meshtype
studyPath='Results/studyTst/miniset/'

xmax=1e-4
sigma=np.ones(3)


vMode=False
showGraphs=False
generate=True
saveGraphs=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))


study=xCell.SimStudy(studyPath,bbox)

l0Min=1e-6
rElec=1e-6

lastNumEl=0
meshTypes=["adaptive","uniform"]


if generate:
   
    # for var in np.linspace(0.1,0.7,15):
    for maxdepth in range(2,10):
        for meshtype in meshTypes:
        # for maxdepth in range(2,10):
            # if meshtype=='uniform':
            #     maxdepth=var
            # else:
            l0Param=2**(-maxdepth*0.2)
            # l0Param=0.2
            
            setup=study.newSimulation()
            setup.mesh.elementType='Admittance'
            setup.meshtype=meshtype
            setup.mesh.minl0=2*xmax/(2**maxdepth)
            setup.ptPerAxis=1+2**maxdepth
            
            if vMode:
                setup.addVoltageSource(1,np.zeros(3),rElec)
                srcMag=1.
                srcType='Voltage'
            else:
                srcMag=4*np.pi*sigma[0]*rElec
                setup.addCurrentSource(srcMag,np.zeros(3),rElec)
                srcType='Current'

            if meshtype=='uniform':
                newNx=int(np.ceil(lastNumEl**(1/3)))
                nX=newNx+newNx%2
                setup.makeUniformGrid(newNx+newNx%2)
                print('uniform, %d per axis'%nX)
            else:
                def metric(coord,l0Param=l0Param):
                    r=np.linalg.norm(coord)
                    val=l0Param*r #1/r dependence
                    # val=(l0Param*r**2)**(1/3) #current continuity
                    # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
                    # if val<rElec:
                    #     val=rElec
                    
                    if (r+val)<rElec:
                        val=rElec/2
                    return val
                
                # def metric(coord):
                #     r=np.linalg.norm(coord)
                #     # val=l0Param*r #1/r dependence
                #     val=(1e-7*r**2)**(1/3) #current continuity
                #     # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
                #     # if val<rElec:
                #     #     val=rElec
                    
                #     if (r+val)<rElec:
                #         val=rElec/2
                #     return val
                
                setup.makeAdaptiveGrid(metric,maxdepth)
            
        
            # setup.startTiming("Make elements")
      
            coords=setup.mesh.nodeCoords
            
            def boundaryFun(coord):
                r=np.linalg.norm(coord)
                return rElec/(r*np.pi*4)
            
                
            # setup.logTime()

            setup.finalizeMesh()

            # setup.insertSourcesInMesh()
            setup.setBoundaryNodes(boundaryFun)
            # setup.getEdgeCurrents()

            # v=setup.solve()
            v=setup.iterativeSolve(None,1e-9)
            
            setup.getMemUsage(True)
            errEst,_,_,_=setup.calculateErrors()#srcMag,srcType,showPlots=showGraphs)
            print('error: %g'%errEst)
            
            study.newLogEntry(['Error','k','Depth'],[errEst,l0Param,maxdepth])
            study.saveData(setup)
            lastNumEl=len(setup.mesh.elements)
            
            
            # ax=xCell.new3dPlot( bbox)
            # xCell.showEdges(ax, coords, setup.mesh.edges)
            # break
        
            # fig=plt.figure()
            # xCell.centerSlice(fig, setup)
            
            # if saveGraphs:
            #     study.makeStandardPlots()
    

# aniGraph=study.animatePlot(xCell.error2d,'err2d')
# aniGraph=study.animatePlot(xCell.ErrorGraph,'err2d_adaptive',["Mesh type"],['adaptive'])
# aniGraph2=study.animatePlot(xCell.error2d,'err2d_uniform',['Mesh type'],['uniform'])
# aniImg=study.animatePlot(xCell.centerSlice,'img_mesh')
# aniImg=study.animatePlot(xCell.SliceSet,'img_adaptive',["Mesh type"],['adaptive'])
# aniImg2=study.animatePlot(xCell.centerSlice,'img_uniform',['Mesh type'],['uniform'])



# fig=plt.figure()
# plotters=[xCell.ErrorGraph,
#           xCell.SliceSet,
#           xCell.CurrentPlot,
#           xCell.CurrentPlot]
# ptype=['ErrorGraph',
#        'SliceSet',
#        'CurrentShort',
#        'CurrentLong']

# for mt in meshTypes:
#     for ii,p in enumerate(plotters):
#         plt.clf()
#         if ii==3:
#             plotr=p(fig,study,fullarrow=True)
#         else:
#             plotr=p(fig,study)
        
#         plotr.getStudyData(filterCategories=["Mesh type"],
#                           filterVals=[mt])
        
#         name=ptype[ii]+'_'+mt
        
#         ani=plotr.animateStudy(name)
        
        



plotE=xCell.ErrorGraph(plt.figure(),study)#,{'showRelativeError':True})
plotE.getStudyData(filterCategories=["Mesh type"],
                          filterVals=['adaptive'])
aniE=plotE.animateStudy()


plotS=xCell.SliceSet(plt.figure(),study)
plotS.getStudyData(filterCategories=["Mesh type"],
                          filterVals=['adaptive'])
aniS=plotS.animateStudy()


plotC=xCell.CurrentPlot(plt.figure(),study)
plotC.getStudyData(filterCategories=["Mesh type"],
                          filterVals=['adaptive'])
aniC=plotC.animateStudy()

xCell.groupedScatter(study.studyPath+'log.csv',xcat='Number of elements',ycat='Error',groupcat='Mesh type')
nufig=plt.gcf()
study.savePlot(nufig, 'AccuracyCost', '.eps')
study.savePlot(nufig, 'AccuracyCost', '.png')