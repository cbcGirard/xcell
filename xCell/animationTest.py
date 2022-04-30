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
datadir='/home/benoit/smb4k/ResearchData/Results/studyTst/'#+meshtype
# studyPath=datadir+'post-renumber/'
studyPath=datadir+'Boundary_large/rubik0'

xmax=1e-2
sigma=np.ones(3)


vMode=False
showGraphs=False
generate=True
saveGraphs=False

vsrc=1.
isrc=vsrc*4*np.pi*sigma*1e-6

bbox=np.append(-xmax*np.ones(3),xmax*np.ones(3))

# bbox=bbox+np.tile()

study=xCell.SimStudy(studyPath,bbox)

l0Min=1e-6
rElec=1e-6

lastNumEl=0
lastNx=0

# tstVals=['adaptive']
# tstVals=["adaptive","uniform"]
# elementType='Admittance'
# tstVals=['adaptive','equal elements',r'equal $l_0$']
# tstCat='Mesh type'


# tstVals=[False, True]
# tstCat='Vsrc?'

# # tstVals=['Admittance','FEM']
# tstVals=['Admittance','Face']
# # tstVals=['Admittance']
# tstCat='Element type'

# tstVals=[None]
# tstCat='Power'

# tstVals=['Analytic','Ground']
tstVals=['Analytic','Ground','Rubik0']
tstCat='Boundary'

if generate:

    # for var in np.linspace(0.1,0.7,15):
    for meshnum,maxdepth in enumerate(range(3,18)):
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
                srcMag=4*np.pi*sigma[0]*rElec
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
                metric=xCell.makeExplicitLinearMetric(maxdepth,
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

            if tstVal=='Analytic':
                setup.setBoundaryNodes(boundaryFun)
            elif tstVal=='Ground':
                setup.setBoundaryNodes()
            elif tstVal=='Rubik0':
                setup.setBoundaryNodes(expand=True,sigma=1.)
            # setup.getEdgeCurrents()

            # v=setup.solve()
            v=setup.iterativeSolve(None,1e-9)

            setup.applyTransforms()

            setup.getMemUsage(True)
            errEst,err,ana,_,_=setup.calculateErrors()
            FVU=xCell.misc.FVU(ana, err)


            print('error: %g'%errEst)


            # tstVal=setup.getPower()
            # print('power: '+str(tstVal))


            study.newLogEntry(['Error','FVU',tstCat],[errEst,FVU,tstVal])
            study.saveData(setup)
            if meshtype=='adaptive':
                lastNumEl=len(setup.mesh.elements)

                lastNx=setup.ptPerAxis-1

