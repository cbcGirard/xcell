#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:22:58 2021
Main API for handling extracellular simulation
@author: benoit
"""

import numpy as np
import numba as nb
from numba import int64, float64
import math
import scipy
from scipy.sparse.linalg import spsolve, cg
from Visualizers import *
from util import *
import time
import os
import resource 
import pickle

import matplotlib.ticker as tickr
import matplotlib.pyplot as plt


nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0



# @nb.experimental.jitclass([
#     ('value',float64),
#     ('coords',float64[:]),
#     ('radius',float64)
#      ])
class CurrentSource:
    def __init__(self,value,coords,radius=0):
        self.value=value
        self.coords=coords
        self.radius=radius

# @nb.experimental.jitclass([
#     ('value',float64),
#     ('coords',float64[:]),
#     ('radius',float64)
#      ])
class VoltageSource:
    def __init__(self,value,coords,radius=0):
        self.value=value
        self.coords=coords
        self.radius=radius

@nb.experimental.jitclass([
    ('origin', float64[:]),
    ('extents',float64[:]),
    ('l0',float64),
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class Element:
    def __init__(self,origin, extents,sigma):
        self.origin=origin
        self.extents=extents
        self.l0=np.prod(extents)**(1/3)
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoordsRecursively(self):
        coords=np.empty((8,3))
        for ii in range(8):
            weights=np.array([(ii>>n)&1 for n in range(3)],dtype=np.float64)
            offset=self.origin+self.extents*weights
            coords[ii]=offset

        return coords
    
    def getConductanceVals(self):
        pass
    
    def getConductanceIndices(self):
        pass
    
    def getCharLength(self):
        return math.pow(np.prod(self.extents),1.0/3)
    
    def setGlobalIndices(self,indices):
        self.globalNodeIndices=indices

@nb.experimental.jitclass([
    ('origin', float64[:]),
    ('extents',float64[:]),
    ('l0', float64),
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class FEMHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
        self.l0=np.prod(extents)**(1/3)
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoordsRecursively(self):
        coords=np.empty((8,3))
        for ii in range(8):
            weights=np.array([(ii>>n)&1 for n in range(3)],dtype=np.float64)
            offset=self.origin+self.extents*weights
            coords[ii]=offset

        return coords
        
    def getCharLength(self):
        return math.pow(np.prod(self.extents),1.0/3)
    
    def setGlobalIndices(self,indices):
        self.globalNodeIndices=indices
        
        
    def getConductanceVals(self):
        if self.sigma.shape[0]==1:
            sigma=self.sigma*np.ones(3)
        else:
            sigma=self.sigma
            
        # k=self.extents/(36*np.roll(self.extents,1)*np.roll(self.extents,2))
        k=np.roll(self.extents,1)*np.roll(self.extents,2)/(36*self.extents)
        K=sigma*k
        
        g=np.empty(28,dtype=np.float64)
        nn=0
        weights=np.empty(3,dtype=np.float64)
        for ii in range(8):
            for jj in range(ii+1,8):
                dif=np.bitwise_xor(ii,jj)
            
                mask = np.array([(dif >> i)&1 for i in range(3)])
                numDif = np.sum(mask)
        
                if numDif == 1:
                    coef = 2*(mask^1)-4*mask
                elif numDif == 2:
                    coef = (mask^1)-2*mask
                else:
                    coef = -mask
                    
                weights=-coef.astype(np.float64)
                g0=np.dot(K,weights)
                g[nn]=g0
                nn=nn+1
                
        return g
    
    def getConductanceIndices(self):
        edges=np.empty((28,2),dtype=np.int64)
        nn=0
        for ii in range(8):
            for jj in range(ii+1,8):
                edges[nn,:]=self.globalNodeIndices[np.array([ii,jj])]
                nn+=1
                
        return edges
                
    
    
@nb.experimental.jitclass([
    ('origin', float64[:]),
    ('extents',float64[:]),
    ('l0',float64),
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class AdmittanceHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
        self.l0=np.prod(extents)**(1/3)
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoordsRecursively(self):
        coords=np.empty((8,3))
        for ii in range(8):
            weights=np.array([(ii>>n)&1 for n in range(3)],dtype=np.float64)
            offset=self.origin+self.extents*weights
            coords[ii]=offset

        return coords
    
    def getMidpoint(self):
        return self.origin+self.extents/2
        
    def getCharLength(self):
        return math.pow(np.prod(self.extents),1.0/3)
    
    def setGlobalIndices(self,indices):
        self.globalNodeIndices=indices
        
        
    def getConductanceVals(self):
        if self.sigma.shape[0]==1:
            sigma=self.sigma*np.ones(3)
        else:
            sigma=self.sigma
            
        k=np.roll(self.extents,1)*np.roll(self.extents,2)/self.extents
        K=sigma*k/4
        
        g=np.array([K[ii] for ii in range(3) for jj in range(4)])
        

                
        return g
    
    def getConductanceIndices(self):
        nodesA=np.array([0,2,4,6,0,1,4,5,0,1,2,3])
        offsets=np.array([2**np.floor(ii/4) for ii in range(12)])
        offsets=offsets.astype(np.int64)
        
        # edges=np.array([[self.globalNodeIndices[a],self.globalNodeIndices[a+o]] for a,o in zip(nodesA,offsets)])
        edges=np.empty((12,2),dtype=np.int64)
        for ii in range(12):
            nodeA=nodesA[ii]
            nodeB=nodeA+offsets[ii]
            edges[ii,0]=self.globalNodeIndices[nodeA]
            edges[ii,1]=self.globalNodeIndices[nodeB]
        return edges

# @nb.experimental.jitclass([
#     ('iSourceCoords',float64[:,:]),
#     ('iSourceVals',float64[:]),
#     ])



class Simulation:
    def __init__(self,name,bbox):
        
        self.currentSources=[]
        self.voltageSources=[]
        
        self.vSourceNodes=[]
        self.vSourceVals=[]
        
        
        self.nodeRoleTable=np.empty(0)
        self.nodeRoleVals=np.empty(0)
        
        self.mesh=Mesh(bbox)
        self.currentTime=0.
        
        self.stepLogs=[]
        self.stepTime=[]
        self.memUsage=0
        
        self.nodeVoltages=np.empty(0)
        self.edges=[[]]
        
        self.gMat=[]
        self.RHS=[]
        self.nDoF=0
        
        self.name=name
        self.meshtype='uniform'
        
        self.ptPerAxis=0
        
        self.iteration=0
        
        
    def makeAdaptiveGrid(self,metric,maxdepth):
        self.startTiming("Make elements")
        self.ptPerAxis=2**maxdepth+1
        self.meshtype='adaptive'
        self.mesh=Octree(self.mesh.bbox,maxdepth)
    
        self.mesh.refineByMetric(metric)
        self.logTime()
        
    def makeUniformGrid(self,nX,sigma=np.array([1.,1.,1.])):
        self.meshtype='uniform'
        self.startTiming("Make elements")

        xmax=self.mesh.extents[0]
        self.ptPerAxis=nX+1
           
        xx=np.linspace(-xmax,xmax,nX+1)
        XX,YY,ZZ=np.meshgrid(xx,xx,xx)
        
        
        coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()
        r=np.linalg.norm(coords,axis=1)
        
        self.mesh.nodeCoords=coords
        # self.mesh.extents=2*xmax*np.ones(3)
        
        elOffsets=np.array([1,nX+1,(nX+1)**2])
        nodeOffsets=np.array([np.dot(toBitArray(i),elOffsets) for i in range(8)])
        elExtents=self.mesh.extents/nX
        
        
        
        for zz in range(nX):
            for yy in range(nX):
                for xx in range(nX):
                    elOriginNode=xx+yy*(nX+1)+zz*(nX+1)**2
                    origin=coords[elOriginNode]
                    elementNodes=elOriginNode+nodeOffsets
                    
                    self.mesh.addElement(origin, elExtents, sigma,elementNodes)
                    
        self.logTime()
        print("%d elements in mesh"%(nX**3))
        
    def startTiming(self,stepName):
        """
        General call to start timing an execution step

        Parameters
        ----------
        stepName : string
            Label for the step

        Returns
        -------
        None.

        """
        self.stepLogs.append(Logger(stepName))
        
    def logTime(self):
        """
        Signals completion of step

        Returns
        -------
        None.

        """
        self.stepLogs[-1].logCompletion()
        
    def getMemUsage(self, printVal=False):
        """
        Gets memory usage of simulation

        Returns
        -------
        mem : int
            Platform-dependent, often kb used.

        """
        mem=0
        for log in self.stepLogs:
            mem=max(mem,log.memory)

        
        if printVal:
            engFormat=tickr.EngFormatter(unit='b')
            print(engFormat(mem*1024)+" used")
        
        return mem
    
    

    def getEdgeCurrents(self):
        condMat=scipy.sparse.tril(self.gMat,-1)
        
        edgeMat=np.array(condMat.nonzero()).transpose()
        dofNodes=np.argwhere(self.nodeRoleTable==0).squeeze()
        nDoF=len(dofNodes)
        
        currentNodes=self.nodeRoleTable==2
        nCurrent=sum(currentNodes)
   
        
        dofCurr=-1-np.arange(nCurrent)
        dof2Global=np.concatenate((dofNodes,dofCurr))
        
        srcCoords=np.array([s.coords for s in self.currentSources])
        gCoords=self.mesh.nodeCoords
        
        vvec=self.nodeVoltages
        
        gList=-condMat.data
        
        # currents=[]
        # edges=[]
        # for ii in nb.prange(len(gList)):
        #     g=gList[ii]
        #     globalEdge=dof2Global[edgeMat[ii]]
            
        #     dv=np.diff(vvec[globalEdge])
            
        #     currents.append(dv*g)
        #     edges.append(globalEdge)
            
        currents, edges = edgeCurrentLoop(gList, edgeMat, 
                                          dof2Global, vvec,
                                          gCoords, srcCoords)
        return currents, edges
    
        
    
    def intifyCoords(self):
        nx=self.ptPerAxis-1
        bb=self.mesh.bbox
        
        span=bb[3:]-bb[:3]
        float0=self.mesh.nodeCoords-bb[:3]
        ints=np.rint((nx*float0)/span)
        
        return ints.astype(np.int64)
    
    
    def makeTableHeader(self):
        cols=[
            "File name",
            "Mesh type",
            "Domain size",
            "Element type",
            "Number of nodes",
            "Number of elements",
            ]
        
        for log in self.stepLogs:
            cols.append(log.name)
            
        cols.extend(["Total time","Max memory"])
        return ','.join(cols)
    
    def logAsTableEntry(self,csvFile,extraCols=None, extraVals=None):
        """
        Logs key metrics of simulation as an additional line of a .csv file.
        Custom categories (column headers) and their values can be added to the line

        Parameters
        ----------
        csvFile : file path
            File where data is written to.
        extraCols : string[:], optional
            Additional categories (column headers). The default is None.
        extraVals : numeric[:], optional
            Values corresponding to the additional categories. The default is None.

        Returns
        -------
        None.

        """
        oldfile=os.path.exists(csvFile)
        f=open(csvFile,'a')
        
        if not oldfile:
            f.write(self.makeTableHeader())
            
            if extraCols is not None:
                f.write(','+','.join(extraCols))
                
            f.write('\n')
        
        
        
        data=[
            self.name,
            self.meshtype,
            np.mean(self.mesh.extents),
            self.mesh.elementType,
            self.mesh.nodeCoords.shape[0],
            len(self.mesh.elements),
            ]
        dt=0
        memory=0
        for log in self.stepLogs:
            dt+=log.duration
            data.append(log.duration)
            memory=max(memory,log.memory)
            
        data.append(dt)
        
        f.write(','.join(map(str,data)))
        f.write(','+str(memory))
        if extraVals is not None:
            f.write(','+','.join(map(str,extraVals)))
        
        f.write('\n')
        f.close()
        
    def finalizeMesh(self):
        self.mesh.finalize()
        numEl=len(self.mesh.elements)
        
        print('%d elem'%numEl)
        nNodes=len(self.mesh.nodeCoords)
        self.nodeRoleTable=np.zeros(nNodes,dtype=np.int64)
        self.nodeRoleVals=np.zeros(nNodes,dtype=np.int64)
        # self.insertSourcesInMesh()
        
    def addCurrentSource(self,value,coords,radius=0):
        self.currentSources.append(CurrentSource(value,coords,radius))

    def addVoltageSource(self,value,coords=None,radius=0):
        self.voltageSources.append(VoltageSource(value,coords,radius))
        
    def insertSourcesInMesh(self,snaplength=0):
        for ii in nb.prange(len(self.voltageSources)):
            src=self.voltageSources[ii]
            
            indices=self.__nodesInSource(src)
            
            self.nodeRoleTable[indices]=1
            self.nodeRoleVals[indices]=ii
            
            # self.vSourceNodes.extend(indices)
            # self.vSourceVals.extend(src.value*np.ones(len(indices)))
            self.nodeVoltages[indices]=src.value
            
        for ii in nb.prange(len(self.currentSources)):
            src=self.currentSources[ii]
            
            indices=self.__nodesInSource(src)
            
            self.nodeRoleTable[indices]=2
            self.nodeRoleVals[indices]=ii      
    
    
    def __nodesInSource(self, source):
        
        
        d=np.linalg.norm(source.coords-self.mesh.nodeCoords,axis=1)
        inside=d<=source.radius
        
        if sum(inside)>0:
            # index=self.mesh.indexMap[inside]
            index=np.nonzero(inside)[0]

        else:
            el=self.mesh.getContainingElement(source.coords)
            elIndices=sparse2denseIndex(el.globalNodeIndices, self.mesh.indexMap)
            elCoords=self.mesh.nodeCoords[elIndices]
            
            d=np.linalg.norm(source.coords-elCoords,axis=1)
            index=elIndices[d==min(d)]
            
       
        return index
            
    def setBoundaryNodes(self,boundaryFun=None):
        
        bnodes=self.mesh.getBoundaryNodes()
        self.nodeVoltages=np.empty(self.mesh.nodeCoords.shape[0])
        
    
        if boundaryFun is None:
            bvals=np.zeros_like(bnodes)
        else:
            bcoords=self.mesh.nodeCoords[bnodes]
            blist=[]
            for ii in nb.prange(len(bnodes)):
                blist.append(boundaryFun(bcoords[ii]))
            bvals=np.array(blist)
            
        self.nodeVoltages[bnodes]=bvals
        self.nodeRoleTable[bnodes]=1
                
    
    def solve(self):
        """
        Directly solves for nodal voltages, given current mesh and distribution
        of sources. Computational time grows significantly with simulation size;
        try iterativeSolve() for faster convergence
        
        Returns
        -------
        voltages : float[:]
            Nodal voltages.

        """
        self.startTiming("Calculate conductances")
        edges,conductances=self.mesh.getConductances()
        self.edges=edges
        self.conductances=conductances
        self.logTime()
        
        self.startTiming("Sort node types")
        nodeType,nodeSubset, vFix2Global, iFix2Global, dof2Global = self.getNodeTypes()
        self.logTime()
        
        
        nNodes=self.mesh.nodeCoords.shape[0]
        voltages=np.empty(nNodes,dtype=np.float64)

        
        b = self.setRHS()
        
        M=self.getMatrix(nNodes,edges,conductances,nodeType,nodeSubset,b,dof2Global)
        
        self.startTiming('Solving')
        vDoF=spsolve(M.tocsc(), b)
        self.logTime()
        
        voltages[dof2Global]=vDoF
        
        self.nodeVoltages=voltages
        return voltages


    def iterativeSolve(self,vGuess=None,tol=1e-5):
        """
        Solves nodal voltages using conjgate gradient method. Likely to achieve
        similar accuracy to direct solution at much greater speed for element
        counts above a few thousand

        Parameters
        ----------
        vGuess : TYPE
            DESCRIPTION. Default None.
        tol : TYPE, optional
            DESCRIPTION. The default is 1e-5.

        Returns
        -------
        voltages : TYPE
            DESCRIPTION.

        """
        self.startTiming("Calculate conductances")
        edges,conductances=self.mesh.getConductances()
        self.edges=edges
        self.conductances=conductances
        self.logTime()
        
        self.startTiming("Sort node types")
        # nodeType,nodeSubset, vFix2Global, iFix2Global, dof2Global = self.getNodeTypes()
        self.getNodeTypes()
        self.logTime()
        
        
        # nNodes=self.mesh.nodeCoords.shape[0]
        voltages=self.nodeVoltages

        # nFixedV=len(vFix2Global)
        
        # dofNodes=np.setdiff1d(range(nNodes), self.vSourceNodes)
        dof2Global=np.nonzero(self.nodeRoleTable==0)[0]
        nDoF=sum(self.nodeRoleTable==0)

        b = self.setRHS(nDoF)
        
        M=self.getMatrix()
        
        self.startTiming('Solving')
        vDoF,_=cg(M.tocsc(),b,vGuess,tol)
        self.logTime()
        
        
        voltages[dof2Global]=vDoF[:nDoF]
        
        # for nn in range(len(self.currentSources)):
        #     v=vDoF[nDoF+nn]
        #     voltages[np.logical_and(self.nodeRoleTable==2,self.nodeRoleVals==nn)]=v
        
        for nn in range(nDoF,len(vDoF)):
            sel=self.__selByDoF(nn)
            voltages[sel]=vDoF[nn]
        
        self.nodeVoltages=voltages
        return voltages
    
    def __selByDoF(self, dofNdx):
        nDoF=sum(self.nodeRoleTable==0)
        
        nCur=dofNdx-nDoF
        if nCur<0:
            selector=np.zeros_like(self.nodeRoleTable,dtype=bool)
        else:
            selector=np.logical_and(self.nodeRoleTable==2,self.nodeRoleVals==nCur)
        
        return selector
        
    
    def calculateErrors(self):
        
        srcV=[]
        srcI=[]
        srcLocs=[]
        srcRadii=[]
        
        for ii in nb.prange(len(self.currentSources)):
            I=self.currentSources[ii].value
            rad=self.currentSources[ii].radius
            srcI.append(I)
            srcLocs.append(self.currentSources[ii].coords)
            srcRadii.append(rad)
            
            if rad>0:
                V=I/(4*np.pi*rad)
            srcV.append(V)
            
        for ii in nb.prange(len(self.voltageSources)):
            V=self.voltageSources[ii].value
            srcV.append(V)
            srcLocs.append(self.voltageSources[ii].coords)
            rad=self.voltageSources[ii].radius
            srcRadii.append(rad)
            if rad>0:
                I=V*4*np.pi*rad
                
            srcI.append(I)
            
        def __analytic(rad,V,I,r):
            inside=r<rad
            voltage=np.empty_like(r)
            voltage[inside]=V
            voltage[~inside]=I/(4*np.pi*r[~inside])
            return voltage
            
        # nTypes,_,_,_,_=self.getNodeTypes()
        v=self.nodeVoltages
        
        vAna=np.zeros_like(v)
        anaInt=0.
        
        coords=self.mesh.nodeCoords
        
        for ii in nb.prange(len(srcI)):
            
            r=np.linalg.norm(coords-srcLocs[ii],axis=1)
            rDense=np.linspace(0,max(r),100)
            
            # inside=r<srcRadii[ii]
            
            # vEst=np.empty_like(v)
            # vEst[inside]=srcV[ii]
            # vEst[~inside]=srcI[ii]/(4*np.pi*r[~inside])

            vEst=__analytic(srcRadii[ii], srcV[ii], srcI[ii], r)
            vDense=__analytic(srcRadii[ii], srcV[ii], srcI[ii], rDense)
            vAna+=vEst
            anaInt+=np.trapz(vDense,rDense)
            
        sorter=np.argsort(np.linalg.norm(coords,axis=1))
        rsort=r[sorter]
        
            
        err=v-vAna
        errSort=err[sorter]
        errSummary=np.trapz(abs(errSort),rsort)/anaInt


        return errSort, errSummary
    
    
    def setRHS(self,nDoF):
        """
        Set right-hand-side of system of equations (as determined by simulation's boundary conditions)

        Parameters
        ----------
        nDoF : int64
            Number of degrees of freedom. Equal to number of nodes with unknown voltages plus number of nodes
            with a fixed injected current
        nodeSubset : TYPE
            DESCRIPTION.

        Returns
        -------
        b : float[:]
            RHS of system Gv=b.

        """
        #set right-hand side
        self.startTiming('Setting RHS')
        nCurrent=len(self.currentSources)
        b=np.zeros(nDoF+nCurrent,dtype=np.float64)
        
        # for n,val in zip(self.iSourceNodes,self.iSourceVals):
        #     # b[global2subset(n,nDoF)]=val
        #     nthDoF=nodeSubset[n]
        #     b[nthDoF]=val
        
        for ii in nb.prange(nCurrent):
            b[nDoF+ii]=self.currentSources[ii].value
            
        self.logTime()
        self.RHS=b
        
        return b
    
    
    def getMatrix(self):
        """
        Calculates conductivity matrix G for the current mesh.

        Parameters
        ----------

        Returns
        -------
        G: sparse matrix
            Conductance matrix G in system Gv=b.

        """
        self.startTiming("Filtering conductances")
        #diagonal elements are -sum of row
        
        nCurrent=len(self.currentSources)
        nDoF=sum(self.nodeRoleTable==0)+nCurrent
        
        
        edges=self.edges
        conductances=self.conductances
        b=self.RHS
        
        nodeA=[]
        nodeB=[]
        cond=[]
        for ii in nb.prange(len(conductances)):
            edge=edges[ii]
            g=conductances[ii]
            
            for nn in range(2):
                this=np.arange(2)==nn
                thisrole,thisDoF=self.__toDoF(edge[this][0])
                thatrole,thatDoF=self.__toDoF(edge[~this][0])

                if thisDoF is not None:
                    # valid degree of freedom
                    cond.append(g)
                    nodeA.append(thisDoF)
                    nodeB.append(thisDoF)
                    
                    if thatDoF is not None:
                        # other node is DoF
                        cond.append(-g)
                        nodeA.append(thisDoF)
                        nodeB.append(thatDoF)
                        
                    else:
                        #other node is fixed voltage
                        thatVoltage=self.nodeVoltages[edge[~this]]
                        b[thisDoF]-=g*thatVoltage
      
        self.logTime()
    
    
        self.startTiming("assembling system")

        G=scipy.sparse.coo_matrix((cond,(nodeA,nodeB)),shape=(nDoF,nDoF))
        G.sum_duplicates()
        
        self.logTime()
        self.gMat=G
        self.RHS=b
        return G

    def __toDoF(self,globalIndex):
        role=self.nodeRoleTable[globalIndex]
        roleVal=self.nodeRoleVals[globalIndex]
        if role==1:
            dofIndex= None
        else:
            dofIndex=roleVal
            if role==2:
                dofIndex+=self.nDoF
                
        return role,dofIndex

    def getNodeTypes(self):
        """
        Gets an integer per node indicating its role:
            0: Unknown voltage
            1: Fixed voltage
            2: Fixed current, unknown voltage

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        self.insertSourcesInMesh()
        
        self.nDoF=sum(self.nodeRoleTable==0)
        trueDoF=np.nonzero(self.nodeRoleTable==0)[0]
        
        for n in nb.prange(len(trueDoF)):
            self.nodeRoleVals[trueDoF[n]]=n
        
    
    def resamplePlane(self,nXSamples=None,normalAxis=2,axisLocation=0.):
        
        xmax=max(self.mesh.extents)

        if nXSamples is None:
            nXSamples=self.ptPerAxis
                    
        coords=self.mesh.nodeCoords
        sel=np.equal(coords[:,normalAxis],axisLocation)
        selAx=np.array([n for n in range(3) if n!=normalAxis])
            
        planeCoords=coords[:,selAx]
        sliceCoords=planeCoords[sel]
        vPlane=self.nodeVoltages[sel]
        
        # kX=self.ptPerAxis//2
        kX=nXSamples//2
        sliceIdx=np.array(kX+kX*sliceCoords/(xmax*np.ones(2)),dtype=np.int64)
        
        
        xx=np.linspace(-xmax,xmax,nXSamples)
        XX,YY=np.meshgrid(xx,xx)
        
        interpCoords=np.vstack((XX.ravel(),YY.ravel(),np.zeros_like(XX.ravel()))).transpose()
        vInterp=np.empty_like(XX)
        
        #set known values
        for ii in nb.prange(len(vPlane)):
            x,y=sliceIdx[ii]
            vInterp[x,y]=vPlane[ii]
        
        print('%d of %d known'%(len(vPlane),nXSamples**2))
            
        
        for ii in nb.prange(nXSamples):
            sameX=sliceIdx[:,0]==ii
            # print('%d of %d'%(ii,nXSamples))

            for jj in nb.prange(nXSamples):
                sameY=sliceIdx[:,1]==jj
                xyMatch=np.logical_and(sameX,sameY)
                
                if not any(xyMatch):
                    x0=maxUnder(ii,sliceIdx[:,0])
                    x1=minOver(ii,sliceIdx[:,0])
                    y0=maxUnder(jj,sliceIdx[:,1])
                    y1=minOver(jj,sliceIdx[:,1])
                    
                    vNodes=np.array([vInterp[x,y] for y in [y0,y1] for x in [x0,x1]])
                    
                    local=np.zeros(2,dtype=np.float64)
                    if x0!=x1:
                        local[0]=(ii-x0)/(x1-x0)
                    if y1!=y0:
                        local[1]=(jj-y0)/(y1-y0)
                    
                    
                    
                    vInterp[ii,jj]=interpolateBilin(vNodes,local)
                    # interpCoord=interpCoords[ii*nXSamples+jj]
                    # container=self.mesh.getContainingElement(interpCoord)
                    # localCoord=(interpCoord-container.origin)/container.span
                    
                    # contNodes=container.globalNodeIndices
                    # meshNodes=sparse2denseIndex(contNodes,self.mesh.indexMap)
                    # contV=self.nodeVoltages[meshNodes]
                    
                    
                    # interp=getElementInterpolant(contV)
                    # vInterp[ii,jj]=evalulateInterpolant(interp, localCoord)
            
        return vInterp, interpCoords
    
                
def getMappingArrays(generalArray,subsetArray):
    gen2sub=-np.ones(len(generalArray))
    sub2gen=[np.argwhere(n==generalArray).squeeze() for n in subsetArray]
    
    return gen2sub, sub2gen
        
class Mesh:
    def __init__(self,bbox,elementType='Admittance'):
        self.bbox=bbox
        self.extents=(bbox[3:]-bbox[:3])/2
        self.elements=[]
        self.conductances=[]
        self.elementType=elementType
        self.nodeCoords=np.empty((0,3),dtype=np.float64)
        self.edges=[]
        
        self.minl0=0
        
        
    def __getstate__(self):
        
        state=self.__dict__.copy()
        
        elInfo=[]
        for el in self.elements:
            d={'origin':el.origin,
               'extents':el.extents,
               'l0':el.l0,
               'sigma':el.sigma,
               'nodeIndices':el.globalNodeIndices}
            elInfo.append(d)
            
        state['elements']=elInfo
        return state
    
    def __setstate__(self,state):
        self.__dict__.update(state)
        elDicts=self.elements.copy()
        self.elements=[]
        
        for ii,el in enumerate(elDicts):
            self.addElement(el['origin'], el['extents'],
                            el['sigma'], el['nodeIndices'])
        
    def getContainingElement(self,coords):
        """
        Get element containing specified point

        Parameters
        ----------
        coords : float[:]
            Cartesian coordinates of point.

        Raises
        ------
        ValueError
            Error if no element contains the point.

        Returns
        -------
        elem : TYPE
            Containing element.

        """
        nElem=len(self.elements)
        
        #TODO: workaround fudge factor
        tol=1e-9*np.ones(3)
        
        for nn in nb.prange(nElem):
            elem=self.elements[nn]
            # if type(elem) is dict:
            #     delta=coords-elem['origin']
            #     ext=elem['extents']
            # else:
            delta=coords-elem.origin+tol
            ext=elem.extents+2*tol
                
                
            difs=np.logical_and(delta>=0,delta<=ext)
            if all(difs):
                return elem
           
        raise ValueError('Point (%s) not inside any element' % ','.join(map(str,coords)))
            
       
    def finalize(self):
       pass
       
    def addElement(self,origin, extents,sigma,nodeIndices):
        """
        Inserts element into the mesh

        Parameters
        ----------
        origin : float[:]
            Cartesian coords of element's origin.
        extents : float[:]
            Length of edges in x,y,z.
        sigma : float
            Conductivity of element.
        nodeIndices : int64[:]
            Numbering of element nodes according to global mesh.

        Returns
        -------
        None.

        """
        if self.elementType=='Admittance':
            newEl=AdmittanceHex(origin,extents,sigma)
        elif self.elementType=='FEM':
            newEl=FEMHex(origin,extents,sigma)
            
        newEl.setGlobalIndices(nodeIndices)
        self.elements.append(newEl)
        
    def getConductances(self):
        """
        Gets the discrete conductances from every element

        Returns
        -------
        edgeIndices : int64[:,:]
            List of node pairs spanned by each conductance.
        conductances : float
            Conductance in siemens.

        """
        nElem=len(self.elements)
        if self.elementType=='Admittance':
            nElemEdge=12
        elif self.elementType=='FEM':
            nElemEdge=28
            
        nEdges=nElemEdge*nElem
        
        conductances=np.empty(nEdges,dtype=np.float64)
        edgeIndices=np.empty((nEdges,2),dtype=np.int64)
        
        for nn in nb.prange(nElem):
            
            elem=self.elements[nn]
            elConds=elem.getConductanceVals()
            elEdges=elem.getConductanceIndices()
            conductances[nn*nElemEdge:(nn+1)*nElemEdge]=elConds
            edgeIndices[nn*nElemEdge:(nn+1)*nElemEdge,:]=elEdges
        
        self.edges=edgeIndices
        return edgeIndices,conductances
    
    def getL0Min(self):
        """
        Get the smallest edge length in mesh

        Returns
        -------
        l0Min : float
            smallest edge length.

        """
        l0Min=np.infty
        
        for el in self.elements:
            l0Min=min(l0Min,el.l0)
        return l0Min

    def getBoundaryNodes(self):
        mins,maxes=np.hsplit(self.bbox,2)
        
        atmin=np.equal(mins,self.nodeCoords)
        atmax=np.equal(maxes,self.nodeCoords)
        isbnd=np.any(np.logical_or(atmin,atmax),axis=1)
        globalIndices=np.nonzero(isbnd)[0]
        
        # globalIndices=sparse2denseIndex(tmpIndex,self.indexMap)
    
        return globalIndices

class Logger():
    def __init__(self,stepName,printStart=True):
        self.name=stepName
        if printStart:
            print(stepName+" starting")
        self.start=time.process_time()
        self.duration=0
        self.memory=0

        
    def logCompletion(self):
        tend=time.process_time()
        duration=tend-self.start
        engFormat=tickr.EngFormatter()
        print(self.name+": "+engFormat(duration)+ " seconds")
        self.duration=duration       
        self.memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        

class Octree(Mesh):
    def __init__(self,boundingBox,maxDepth=10,elementType='Admittance'):
        self.center=np.mean(boundingBox.reshape(2,3),axis=0)
        self.span=(boundingBox[3:]-boundingBox[:3])
        super().__init__(boundingBox, elementType)

        self.maxDepth=maxDepth
        self.bbox=boundingBox
        self.indexMap=np.empty(0,dtype=np.int64)
        
        coord0=self.center-self.span/2
        
        self.tree=Octant(coord0,self.span)
        
    def getContainingElement(self, coords):
        
        el=self.tree.getContainingElement(coords)
        return el
        
        
    def refineByMetric(self,l0Function):
        """
        Recursively splits elements until l0Function evaluated at the center
        of each element is greater than that element's l0'

        Parameters
        ----------
        l0Function : function
            Function returning a scalar for each input cartesian coordinate.

        Returns
        -------
        None.

        """
        self.tree.refineByMetric(l0Function, self.maxDepth)

            
#TODO: vectorize
    def coord2Index(self,coord):
        x0=self.center-self.span/2
        nE=2**(self.maxDepth)
        dX=self.span/(nE)
        
        ndxOffsets=np.array([(nE+1)**n for n in range(3)])
        
        idxArray=(coord-x0)/dX
        newInd=np.dot(idxArray,ndxOffsets)
        
        idx=np.rint(newInd).astype(np.int64)
        if len(self.indexMap)!=0:
            idx=reindex(idx,self.indexMap)
        return idx
        
    def finalize(self):
        octs=self.tree.getTerminalOctants()
        # self.elements=octs
        self.nodeCoords=self.getCoordsRecursively()
        # indices=self.indexMap
        # mesh.nodeCoords=coords
        
        

        for ii in nb.prange(len(octs)):
            o=octs[ii]
            onodes=o.globalNodeIndices
            gnodes=sparse2denseIndex(onodes, self.indexMap)
            self.addElement(o.origin,o.span,np.ones(3),gnodes)
            # o.setGlobalIndices(gnodes)
        
        # T.logCompletion()
        return 
    
    def printStructure(self):
        """
        Debug tool to print structure of tree

        Returns
        -------
        None.

        """
        self.tree.printStructure()
        
    def octantByList(self,indexList,octant=None):
        """
        Selects an octant by recursing through a list of indices

        Parameters
        ----------
        indexList : int64[]
            Octant identifier, where i=indexList[n] specifies child i of octant n
        octant : Octant, optional
            Used internally to recurse. The default is None.

        Returns
        -------
        Octant
            The octant object specified.

        """
        head=indexList.pop(0)
        if octant is None:
            octant=self.tree
        oc=octant.children[head]
        if len(oc.children)==0:
            return oc
        else:
            return self.octantByList(indexList,oc)
        
    def countElements(self):
        
        return self.tree.countElements()
    
    def getCoordsRecursively(self):
        """
        Determines coordinates of mesh nodes

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        coords,i=self.tree.getCoordsRecursively(self.bbox,self.maxDepth)
        
        indices=np.array(i)
        self.indexMap=indices
        self.nodeCoords=coords

        return np.array(coords)
         
    def getBoundaryNodes(self):
        bnodes=[]
        nX=1+2**self.maxDepth
        for ii in nb.prange(len(self.indexMap)):
            nn=self.indexMap[ii]
            xyz=index2pos(nn,nX)
            if np.any(xyz==0) or np.any(xyz==(nX-1)):
                bnodes.append(ii)
        
        return np.array(bnodes)



# @nb.njit(nb.int64[:](nb.int64[:],nb.int64[:]))
@nb.njit()
def sparse2denseIndex(sparseVals,denseVals):
    """
    Gets the indices where each sparseVal are within DenseVals.
    Used for e.g. mapping a global node index to degree of freedom index
    Assumes one-to-one mapping of every value

    Parameters
    ----------
    sparseVals : int64[:]
        List of nonconsecutive values to find.
    denseVals : int64[:]
        List of values to be searched.

    Returns
    -------
    int64[:]
        Indices where each sparseVal occurs in denseVals.

    """
    # idxList=[reindex(sp,denseVals) for sp in sparseVals]
    idxList=[]
    for ii in nb.prange(len(sparseVals)):
        idxList.append(reindex(sparseVals[ii],denseVals))
                         
        
    return np.array(idxList,dtype=np.int64)

@nb.njit()
def reindex(sparseVal,denseList):
    """
    Get position of sparseVal in denseList, returning as soon as found

    Parameters
    ----------
    sparseVal : int64
        Value to find index of match.
    denseList : int64[:]
        List of nonconsecutive indices.

    Returns
    -------
    int64
        index where sparseVal occurs in denseList.

    """
    # startguess=sparseVal//denseList[-1]
    for n,val in np.ndenumerate(denseList):
    # for n,val in enumerate(denseList):

        if val==sparseVal:
            return n[0]
        
# octantspec= [
#     ('origin',nb.float64[:]),
#     ('span',nb.float64[:]),
#     ('center',nb.float64[:]),
#     ('l0',nb.float64),
#     ('children',List[Octant]),
#     ('depth',nb.int64),
#     ('globalNodes',nb.int64[:]),
#     ('nX',nb.int64),
#     ('index',nb.int64),
#     ('nodeIndices',nb.int64[:])
#     ]
# @nb.experimental.jitclass(spec=octantspec)
class Octant():
    def __init__(self,origin, span,depth=0,sigma=np.ones(3),index=[]):
        # super().__init__(origin, span, sigma)
        self.origin=origin
        self.span=span
        self.center=origin+span/2
        self.l0=np.prod(span)**(1/3)
        self.children=[]
        self.depth=depth
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        self.nX=2
        self.index=index        

        
    def calcGlobalIndices(self,globalBbox,maxdepth):
        # x0=globalBbox[:3]
        nX=2**maxdepth
        # dX=(globalBbox[3:]-x0)/(nX)
        # coords=self.getOwnCoords()
        
        toGlobal=np.array([(nX+1)**n for n in range(3)])
        
        # for N,c in enumerate(coords):
        #     idxArray=(c-x0)/dX
        #     ndx=np.dot(ndxOffsets,idxArray)
        #     self.globalNodeIndices[N]=ndx
            
        # return self.globalNodeIndices
        xyz=np.zeros(3,dtype=np.int64)
        for ii,idx in enumerate(self.index):
            ldepth=maxdepth-ii-1
            step=2**ldepth
            xyz+=step*toBitArray(idx)
            
        ownstep=2**(maxdepth-self.depth)
        
        ownXYZ=[xyz+ownstep*toBitArray(i) for i in range(8)]
        indices=np.array([np.dot(ndxlist,toGlobal) for ndxlist in ownXYZ])
            
        self.globalNodeIndices=indices
        return indices
            
            
        
    def countElements(self):
        if len(self.children)==0:
            return 1
        else:
            return sum([ch.countElements() for ch in self.children])
     
    # @nb.njit(parallel=True)
    def makeChildren(self,division=np.array([0.5,0.5,0.5])):
        newSpan=self.span*division
        
        
        for ii in nb.prange(8):
            offset=toBitArray(ii)*newSpan
            newOrigin=self.origin+offset
            newIndex=self.index.copy()
            newIndex.append(ii)
            self.children.append(Octant(newOrigin,newSpan,self.depth+1,index=newIndex))
            
        # return self.children
    def getOwnCoords(self):
        return [self.origin+self.span*toBitArray(n) for n in range(8)]

    
    def getCoordsRecursively(self,bbox,maxdepth):
        # T=Logger('depth  %d'%self.depth, printStart=False)
        if len(self.children)==0:
            # coords=[self.origin+self.span*toBitArray(n) for n in range(8)]
            coords=self.getOwnCoords()
            # indices=self.globalNodeIndices
            indices=self.calcGlobalIndices(bbox, maxdepth)
        else:
            coordList=[]
            indexList=[]
            
            for ch in self.children:
                c,i = ch.getCoordsRecursively(bbox,maxdepth)
                
                if len(coordList)==0:
                    coordList=c
                    indexList=i
                else:
                    coordList.extend(c)
                    indexList.extend(i)
            
            indices,sel=np.unique(indexList,return_index=True)
            
            # self.globalNodeIndices=indices
            
            coords=np.array(coordList)[sel]
            coords=coords.tolist()
                
        # T.logCompletion()
        return coords, indices.tolist()
    
    # def index2pos(self,ndx,dX):
    #     arr=[]
    #     for ii in range(3):
    #         arr.append(ndx%dX)
    #         ndx=ndx//dX
    #     return np.array(arr)
    
    # def pos2index(self,pos,dX):
    #     vals=np.array([dX**n for n in range(3)])
    #     newNdx=np.dot(vals,pos)
    #     return np.rint(newNdx)
    
    
    def getIndicesRecursively(self):
        
        if self.isTerminal():
            return np.arange(8,dtype=np.int64)
        
        else:
            indices=[]
            for ch in self.children:
                indices.append(ch.getIndicesRecursively())
    
            return indices
    
    # @nb.njit(parallel=True)
    def refineByMetric(self,l0Function,maxDepth):
        l0Target=l0Function(self.center)
        # print("target\t%g"%l0Target)
        # print("l0\t\t%g"%self.l0)
        # print(self.center)
        if (self.l0>l0Target) and (self.depth<maxDepth):
            # print('\n\n')
            # print('depth '+str(self.depth)+', child'+str(self.index))

            self.makeChildren()
            for ii in nb.prange(8):
                self.children[ii].refineByMetric(l0Function,maxDepth)
                
    def printStructure(self):
        """
        Prints out octree structure

        Returns
        -------
        None.

        """
        base='> '*self.depth
        print(base+str(self.l0))
        
        for ch in self.children:
            ch.printStructure()
            
    
    def isTerminal(self):
        return len(self.children)==0
        
    def containsPoint(self,coord):
        gt=np.greater_equal(coord,self.origin)
        lt=np.less_equal(coord-self.origin,self.span)
        return np.all(gt&lt)
    
    
    def distributeNodes(self,nodeCoords,nodeIndices):
        if self.isTerminal():
            return [], []
        else:
            for N in len(nodeIndices):
                if self.containsPoint(nodeCoords[N]):
                    self.innerNodes
                    
            
            
            
            return nodeCoords,nodeIndices
        
        
        
    def getTerminalOctants(self):
        """
        Gets all childless octants

        Returns
        -------
        list of Octants
            Childless octants (the actual elements of the mesh)

        """
        if len(self.children)==0:
            return [self]
        else:
            
            descendants=[]
            for ch in self.children:
            # if len(ch.children)==0:
                grandkids=ch.getTerminalOctants()
                
                if grandkids is not None:
                    descendants.extend(grandkids)
        return descendants
    
    
    def getContainingElement(self,coords):
        if len(self.children)==0:
            if self.containsPoint(coords):
                return self
            else:
                return None
        else:
            for ch in self.children:
                tmp=ch.getContainingElement(coords)
                if tmp is not None:
                    return tmp
            return None
                


    

        
        
def arrXor(arr):
    val=False
    for a in arr:
        val=val^a
        
    return a


def analyticVsrc(srcCoord,srcAmplitude,rVec,srcType='Current',sigma=1, srcRadius=1e-6):
    """
    Calculates voltage at an external point, due to current or voltage source

    Parameters
    ----------
    srcCoord : float[:]
        coords of source.
    srcAmplitude : float
        Source amplitude in volts or amperes.
    rVec : float[:]
        Vector of distances from source to evaluate voltage at.
    srcType : 'Current' or 'Voltage', optional
        Type of source. The default is 'Current'.
    sigma : float, optional
        Conductivity of surroundings. The default is 1.
    srcRadius : float, optional
        Effective radius of source, to give an analytical value at its center. The default is 1e-6.

    Returns
    -------
    float[:]
        Voltage at each specified distance.

    """
    r=rVec.copy()
    r[r<srcRadius]=srcRadius

        
    if srcType=='Current':
        v0=srcAmplitude/(4*np.pi*sigma*srcRadius)
    else:
        v0=srcAmplitude
        
    vVec=v0*srcRadius/r
    
            
    return vVec

def getFVU(vsim,analytic,whichPts):
    """
    Calculates fraction of variance unexplained (FVU)

    Parameters
    ----------
    vsim : float[:]
        Simulated nodal voltages.
    analytic : float[:]
        analyitical nodal voltages.
    whichPts : bool[:]
        DESCRIPTION.

    Returns
    -------
    VAF : float
        Variation accounted for
    error : float[:]
        absolute error in simulated solution

    """
    v=vsim[whichPts]
    vA=analytic[whichPts]
    
    delVa=vA-np.mean(vA)
    
    error=v-vA
    
    SSerr=sum(error**2)
    SStot=sum(delVa**2)
    
    FVU=SSerr/SStot
    
    # VAF=np.cov(v,vA)[0,1]**2
    
    return FVU, error

        
class SimStudy:
    def __init__(self,studyPath,boundingBox):
        
        if not os.path.exists(studyPath):
            os.makedirs(studyPath)
        self.studyPath=studyPath
        
        self.nSims=-1
        self.currentSim=None
        self.bbox=boundingBox
        self.span=boundingBox[3:]-boundingBox[:3]
        self.center=boundingBox[:3]+self.span/2
        
        self.iSourceCoords=[]
        self.iSourceVals=[]
        self.vSourceCoords=[]
        self.vSourceVals=[]
        
    def newSimulation(self,simName=None):
        self.nSims+=1
        
        if simName is None:
            simName='sim%d'%self.nSims
            
        sim=Simulation(simName,bbox=self.bbox)
        # sim.mesh.extents=self.span
        
        self.currentSim=sim
        
        return sim
    
    def newLogEntry(self,extraCols=None, extraVals=None):
        fname=os.path.join(self.studyPath,'log.csv')
        self.currentSim.logAsTableEntry(fname,extraCols=extraCols,extraVals=extraVals)
        
        
    def makeStandardPlots(self,savePlots=True,keepOpen=False):
        plotfuns=[error2d, centerSlice]
        plotnames=['err2d','imgMesh']
        
        for f,n in zip(plotfuns,plotnames):
            fig=plt.figure()
            f(fig,self.currentSim)
            
            if savePlots:
                self.savePlot(fig, n, '.png')
                if not keepOpen:
                    plt.close(fig)
        
        
    def saveData(self,simulation):
        data={}
        
        fname=os.path.join(self.studyPath,simulation.name+'.p')
        pickle.dump(simulation,open(fname,'wb'))
        
    def loadData(self,simName):
        fname=os.path.join(self.studyPath,simName+'.p')
        data=pickle.load( open(fname,'rb'))

        return data
        
    def savePlot(self,fig,fileName,ext):
        basepath=os.path.join(self.studyPath,self.currentSim.name)
        
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fname=os.path.join(basepath,fileName+ext)
        plt.savefig(fname)
        
    def loadLogfile(self):
        """
        Returns Pandas dataframe of logged runs

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        cats : TYPE
            DESCRIPTION.

        """
        logfile=os.path.join(self.studyPath,'log.csv')
        df,cats=importRunset(logfile)
        return df,cats
        
    def plotTimes(self,xCat='Number of elements',sortCat=None):
        logfile=os.path.join(self.studyPath,'log.csv')
        df,cats=importRunset(logfile)
        
        if sortCat is not None:
            plotnames=df[sortCat].unique()
        else:
            plotnames=[None]

        for cat in plotnames:
            importAndPlotTimes(logfile,onlyCat=sortCat,onlyVal=cat,xCat=xCat)
            plt.title(cat)
            
    def plotAccuracyCost(self):
        logfile=os.path.join(self.studyPath,'log.csv')
        groupedScatter(logfile, xcat='Total time', ycat='Error', groupcat='Mesh type')
      
        

    def getSavedSims(self,filterCategories=None,filterVals=None,sortCategory=None):
        logfile=os.path.join(self.studyPath,'log.csv')
        df,cats=importRunset(logfile)
            
        if filterCategories is not None:
            selector=np.ones(len(df),dtype=bool)
            for cat,val in zip(filterCategories,filterVals):
                selector&=df[cat]==val
                
            fnames=df['File name'][selector]
        else:
            fnames=df['File name']
            
        if sortCategory is not None:
            sortcats=df[sortCategory]
            
        return fnames
        
        
    def animatePlot(self,plotfun,aniName=None,filterCategories=None,filterVals=None,sortCategory=None):
        # logfile=os.path.join(self.studyPath,'log.csv')
        # df,cats=importRunset(logfile)
            
        # if filterCategories is not None:
        #     selector=np.ones(len(df),dtype=bool)
        #     for cat,val in zip(filterCategories,filterVals):
        #         selector&=df[cat]==val
                
        #     fnames=df['File name'][selector]
        # else:
        #     fnames=df['File name']
            
        # if sortCategory is not None:
        #     sortcats=df[sortCategory]
        fnames=self.getSavedSims(filterCategories=filterCategories,
                                 filterVals=filterVals,
                                 sortCategory=sortCategory)
        
        
        # ims=[]
        fig=plt.figure()
        # loopargs=None
        # # for ii in range(len(fnames)):
        # #     dat=self.loadData(fnames[forder[ii]])
        # for ii,fname in enumerate(fnames):
        #     dat=self.loadData(fname)
        #     im,loopargs=plotfun(fig,dat,loopargs)
        #     # txt=fig.text(0.01,0.95,'frame %d'%ii,
        #     #                horizontalalignment='left',verticalalignment='bottom')
        #     # im.append(txt)
        #     ims.append(im)
        
        plottr=plotfun(fig,self)
        for ii, fname in enumerate(fnames):
            dat=self.loadData(fname)
            plottr.addSimulationData(dat)
            
        ims=plottr.getArtists()
            
            
        ani=mpl.animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=2000,blit=False)
        # ani=mpl.animation.FuncAnimation(fig, aniFun, interval=1000,repeat_delay=2000,blit=True)
        if aniName is None:
            plt.show()
        else:
            ani.save(os.path.join(self.studyPath,aniName+'.mp4'),fps=1)
            
        return ani