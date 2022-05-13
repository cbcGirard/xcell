#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:56:54 2021

@author: benoit
"""

import numpy as np
import numba as nb
import xcell
import matplotlib.pyplot as plt
import pickle

meshtype='adaptive'
# studyPath='Results/studyTst/miniCur/'#+meshtype
datadir='/home/benoit/smb4k/ResearchData/Results/'#+meshtype
# studyPath=datadir+'post-renumber/'

sigma=np.ones(3)

X=1000
V=1
R=2
studyPath=datadir+'errorMetrics/'

generate=True



def run(X,V,R,generate):
    xmax=X*1e-6


    vMode=False

    bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))

    study=xcell.SimStudy(studyPath,bbox)

    l0Min=1e-6
    rElec=R*1e-6

    lastNumEl=0
    lastNx=0

    tstVals=[None]
    tstCat='Power'

    errdicts=[]

    if generate:

        # for var in np.linspace(0.1,0.7,15):
        for meshnum,maxdepth in enumerate(range(3,16)):
            for tstVal in tstVals:
                # meshtype=tstVal
                # elementType=tstVal


                elementType='Admittance'
            # for vMode in tstVals:
            # for maxdepth in range(2,10):
                # if meshtype=='uniform':
                #     maxdepth=var
                # else:
                l0Param=2**(-maxdepth*0.2)
                # l0Param=0.2

                setup=study.newSimulation()
                setup.mesh.elementType=elementType
                setup.meshtype=meshtype
                setup.meshnum=meshnum

                if vMode:
                    setup.addVoltageSource(1,np.zeros(3),rElec)
                    srcMag=1.
                    srcType='Voltage'
                else:
                    srcMag=4*np.pi*sigma[0]*rElec*10
                    setup.addCurrentSource(srcMag,np.zeros(3),rElec)
                    srcType='Current'

                # if meshtype=='equal elements':
                if meshtype=='uniform':
                    newNx=int(np.ceil(lastNumEl**(1/3)))
                    nX=newNx+newNx%2
                    setup.makeUniformGrid(newNx+newNx%2)
                    print('uniform, %d per axis'%nX)
                elif meshtype==r'equal $l_0$':
                    setup.makeUniformGrid(lastNx)
                else:
                    metric=xcell.makeExplicitLinearMetric(maxdepth,
                                                          meshdensity=0.2)

                    setup.makeAdaptiveGrid(metric,maxdepth)


                setup.mesh.elementType=elementType
                setup.finalizeMesh()
                # if asDual:
                #     setup.finalizeDualMesh()
                # else:
                #     setup.finalizeMesh(regularize=False)

                coords=setup.mesh.nodeCoords
                #
                def boundaryFun(coord):
                    r=np.linalg.norm(coord)
                    return rElec/(r*np.pi*4)
                # setup.insertSourcesInMesh()
                setup.setBoundaryNodes(boundaryFun)
                # setup.getEdgeCurrents()

                # v=setup.solve()
                v=setup.iterativeSolve(None,1e-9)

                setup.applyTransforms()

                setup.getMemUsage(True)
                errEst,errVec,vAna,sorter,r=setup.calculateErrors()
                print('error: %g'%errEst)

                sse=np.sum(errVec**2)
                sstot=np.sum((v-np.mean(v))**2)
                ss=np.sum((vAna-np.mean(vAna))**2)
                FVU=sse/sstot

                _,basicAna,basicErr,_=setup.estimateVolumeError(basic=True)
                _,advAna,advErr,_=setup.estimateVolumeError(basic=False)

                errBasic=sum(basicErr)/sum(basicAna)
                errAdv=sum(advErr)/sum(advAna)

                numel=len(setup.mesh.elements)


                power=setup.getPower()
                print('power: '+str(power))
                study.newLogEntry(['AreaError',
                                   'basicVol',
                                   'advVol',
                                   'SSE',
                                   'SStot',
                                   'SS',
                                   'FVU',
                                   'power'],
                                  [errEst,
                                   errBasic,
                                   errAdv,
                                   sse,
                                   sstot,
                                   ss,
                                   FVU,
                                   power])
                study.saveData(setup)

                errdict=xcell.misc.getErrorEstimates(setup)
                # errdict['densities']=density
                errdict['depths']=maxdepth
                errdict['numels']=numel
                errdicts.append(errdict)

        dlist=xcell.misc.transposeDicts(errdicts)
        pickle.dump(dlist,open(studyPath+'errMets.p','wb'))


    else:
        dlist=pickle.load(open(studyPath+'errMets.p','rb'))


    for met in ['FVU','FVU2','powerError','int1','int3']:
        plt.loglog(dlist['numels'],dlist[met],label=met)

    plt.legend()
    plt.xlabel('Number of elements')
    plt.title('%dum domain, %dV-equivalent %dum source'%(X,V,R))



    study.savePlot(plt.gcf(), '%du-%dv-%dr'%(X,V,R), '.png')
    study.savePlot(plt.gcf(), '%du-%dv-%dr'%(X,V,R), '.eps')
    plt.close('all')


for R in [1, 10]:
    for X in [100, 1000, 10000]:
        for V in [1,10]:
            run(X,V,R,generate)
