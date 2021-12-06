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


meshtype='l0/r'
studyPath='Results/studyTst/'+meshtype

maxdepth=6
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

def makeUniformGrid(simulation,nX):
    xmax=simulation.mesh.extents[0]/2
       
    xx=np.linspace(-xmax,xmax,nX+1)
    XX,YY,ZZ=np.meshgrid(xx,xx,xx)
    
    
    coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()
    r=np.linalg.norm(coords,axis=1)
    
    simulation.mesh.nodeCoords=coords
    simulation.mesh.extents=2*xmax*np.ones(3)
    simulation.mesh.elementType='Admittance'
    
    simulation.startTiming("Make elements")
    
    elExtents=np.ones(3)*2*xmax/nX
    elOffsets=np.array([1,nX+1,(nX+1)**2])
    nodeOffsets=np.array([np.dot(xCell.toBitArray(i),elOffsets) for i in range(8)])
    
    
    for zz in range(nX):
        for yy in range(nX):
            for xx in range(nX):
                elOriginNode=xx+yy*(nX+1)+zz*(nX+1)**2
                origin=coords[elOriginNode]
                elementNodes=elOriginNode+nodeOffsets
                
                simulation.mesh.addElement(origin, elExtents, sigma,elementNodes)
                
    simulation.logTime()



if generate:
    # if vMode:
    #     study.vSourceCoords=np.zeros(3)
    #     study.vSourceVals=vsrc
    # else:
    #     study.iSourceCoords=np.zeros(3)
    #     study.iSourceVals=isrc
    if meshtype=='uniform':
        varrange=np.arange(2,7)
    else:
        varrange=np.linspace(1,0.05,10)
    
    
    
    for var in varrange:
        if meshtype=='uniform':
            maxdepth=var
        else:
            l0Param=var
        
        setup=study.newSimulation()
        setup.mesh.elementType='Admittance'
        setup.meshtype=meshtype
        setup.mesh.minl0=2*xmax/(2**maxdepth)
        setup.ptPerAxis=1+2**maxdepth
        
        if meshtype=='uniform':
            makeUniformGrid(setup,2**maxdepth)
            
        else:
            def metric(coord,l0Param=l0Param):
                r=np.linalg.norm(coord)
                val=l0Param*r #1/r dependence
                # val=(l0Param*r**2)**(1/3) #current continuity
                # val=(l0Param*r**4)**(1/3) #dirichlet energy continutity
                if val<rElec:
                    val=rElec
                return val
            otree=xCell.Octree(bbox,maxDepth=maxdepth)
            otree.refineByMetric(metric)
            numEl=len(otree.tree.getTerminalOctants())
            if numEl==lastNumEl:
                continue
            
            setup.mesh=otree.makeMesh(setup.mesh)
        
    
        setup.startTiming("Make elements")
  
        
        
        
        coords=setup.mesh.nodeCoords
        
        
            
        setup.logTime()
        numEl=len(setup.mesh.elements)
        
        print('%d elem'%numEl)
        
        if meshtype=='uniform':
            sourceIndex=coords.shape[0]//2
        else:
            sourceIndex=otree.coord2Index(np.zeros(3))
        
        if vMode:
            srcMag=1.
            srcType='Voltage'
        else:
            srcMag=4*np.pi*sigma[0]*rElec
            setup.addCurrentSource(srcMag,index=sourceIndex)
            srcType='Current'
        
        # ground boundary nodes, set v sources
        for ii in nb.prange(coords.shape[0]):
            pt=np.abs(coords[ii])
            rpt=np.array(np.linalg.norm(pt))
            if np.any(pt==xmax) or (rpt<=rElec):
                # setup.addVoltageSource(0,index=ii)
                
                vpt=xCell.analyticVsrc(np.zeros(3), srcMag, rpt,srcType=srcType,srcRadius=rElec)
                setup.addVoltageSource(vpt.squeeze(),index=ii)
            
        
        
        # v=setup.solve()
        v=setup.iterativeSolve(None,1e-9)
        
        setup.getMemUsage(True)
        FVU=setup.calculateErrors(srcMag,srcType,showPlots=showGraphs)
        print('error: %g'%FVU)
        
        study.newLogEntry(['Areal error'],[FVU])
        study.saveData(setup)
        lastNumEl=numEl
        
        if saveGraphs:
            study.makeStandardPlots()


aniGraph=study.animatePlot(xCell.error2d,'err2d')
aniImg=study.animatePlot(xCell.centerSlice,'img_mesh')


