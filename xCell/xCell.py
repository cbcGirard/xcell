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
# from util import *
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
        """
        Fast utility to construct an octree-based mesh of the domain.

        Parameters
        ----------
        metric : function
            Must take a 1d array of xyz coordinates and return target l0
            for that location.
        maxdepth : int
            Maximum allowable rounds of subdivision.

        Returns
        -------
        None.

        """
        self.startTiming("Make elements")
        self.ptPerAxis=2**maxdepth+1
        self.meshtype='adaptive'
        self.mesh=Octree(self.mesh.bbox,maxdepth)
    
        self.mesh.refineByMetric(metric)
        self.logTime()
        
    def makeUniformGrid(self,nX,sigma=np.array([1.,1.,1.])):
        """
        Fast utility to construct a uniformly spaced mesh of the domain.

        Parameters
        ----------
        nX : int
            Number of elements along each axis (yielding nX**3 total elements).
        sigma : TYPE, optional
            Global conductivity. The default is np.array([1.,1.,1.]).

        Returns
        -------
        None.

        """
        self.meshtype='uniform'
        self.startTiming("Make elements")

        xmax=self.mesh.extents[0]
        self.ptPerAxis=nX+1
           
        xx=np.linspace(-xmax,xmax,nX+1)
        XX,YY,ZZ=np.meshgrid(xx,xx,xx)
        
        
        coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()
        # r=np.linalg.norm(coords,axis=1)
        
        self.mesh.nodeCoords=coords
        # self.mesh.extents=2*xmax*np.ones(3)
        
        elOffsets=np.array([1,nX+1,(nX+1)**2])
        nodeOffsets=np.array([np.dot(util.toBitArray(i),elOffsets) for i in range(8)])
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
        General call to start timing an execution step.

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
        Signals completion of step.

        Returns
        -------
        None.

        """
        self.stepLogs[-1].logCompletion()
        
    def getMemUsage(self, printVal=False):
        """
        Get memory usage of simulation.

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
        """
        Get currents through each edge of the mesh.

        Returns
        -------
        currents : float[:]
            Current through edge in amperes; .
        edges : int[:,:]
            Pairs of node (global) node indices corresponding to 
            [start, end] of each current vector.

        """
        gAll=self.getEdgeMat()
        condMat=scipy.sparse.tril(gAll,-1)
        edges=np.array(condMat.nonzero()).transpose()
        
        dv=np.diff(self.nodeVoltages[edges]).squeeze()
        iTmp=-condMat.data*dv
        
        #make currents positive, and flip direction if negative
        needsFlip=iTmp<0
        currents=abs(iTmp)
        edges[needsFlip]=np.fliplr(edges[needsFlip])
        
        return currents, edges
    
    
    def intifyCoords(self,coords=None):
        """
        Expresses coordinates as triplet of positive integers.
        
        Prevents rounding
        errors when determining if two points correspond to the same
        mesh node
        
        Parameters
        ----------
        coords: float[:,:]
            Coordinates to rescale as integers, or mesh nodes if None.

        Returns
        -------
        int[:,:]
            Mesh nodes as integers.

        """
        nx=self.ptPerAxis-1
        bb=self.mesh.bbox
        
        if coords is None:
            coords=self.mesh.nodeCoords
        
        span=bb[3:]-bb[:3]
        float0=coords-bb[:3]
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
        Log key metrics of simulation as an additional line of a .csv file.
        
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
        
        #TODO: doc better
    def finalizeMesh(self):
        """
        Prepare mesh for simulation.

        Returns
        -------
        None.

        """
        self.mesh.finalize()
        numEl=len(self.mesh.elements)
        
        print('%d elem'%numEl)
        nNodes=len(self.mesh.nodeCoords)
        self.nodeRoleTable=np.zeros(nNodes,dtype=np.int64)
        self.nodeRoleVals=np.zeros(nNodes,dtype=np.int64)
        # self.insertSourcesInMesh()
        
        self.startTiming("Calculate conductances")
        edges,conductances=self.mesh.getConductances()
        self.edges=edges
        self.conductances=conductances
        self.logTime()
        
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
            #TODO: change deprecated
            elIndices=util.sparse2denseIndex(el.globalNodeIndices, self.mesh.indexMap)
            elCoords=self.mesh.nodeCoords[elIndices]
            
            d=np.linalg.norm(source.coords-elCoords,axis=1)
            index=elIndices[d==min(d)]
            
       
        return index
            
    def setBoundaryNodes(self,boundaryFun=None):
        
        bnodes=self.mesh.getBoundaryNodes()
        self.nodeVoltages=np.zeros(self.mesh.nodeCoords.shape[0])
        
    
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
        Directly solves for nodal voltages.
        
        Computational time grows significantly with simulation size;
        try iterativeSolve() for faster convergence
        
        Returns
        -------
        voltages : float[:]
            Simulated nodal voltages.
            
        See Also
        --------
        iterativeSolve: conjugate gradient solver

        """
        self.startTiming("Sort node types")
        self.getNodeTypes()
        self.logTime()
        
        
        # nNodes=self.mesh.nodeCoords.shape[0]
        voltages=self.nodeVoltages

        dof2Global=np.nonzero(self.nodeRoleTable==0)[0]
        nDoF=dof2Global.shape[0]
        
        M,b=self.getSystem()
        self.startTiming('Solving')
        vDoF=spsolve(M.tocsc(), b)
        self.logTime()
        
        voltages[dof2Global]=vDoF[:nDoF]
        
        for nn in range(nDoF,len(vDoF)):
            sel=self.__selByDoF(nn)
            voltages[sel]=vDoF[nn]
        
        self.nodeVoltages=voltages
        return voltages


    def iterativeSolve(self,vGuess=None,tol=1e-5):
        """
        Solves nodal voltages using conjgate gradient method.
        
        Likely to achieve similar accuracy to direct solution at much greater 
        speed for element counts above a few thousand

        Parameters
        ----------
        vGuess : float[:]
            Initial guess for nodal voltages. Default None.
        tol : float, optional
            Maximum allowed norm of the residual. The default is 1e-5.

        Returns
        -------
        voltages : float[:]
            Simulated nodal voltages.

        """

        
        self.startTiming("Sort node types")
        self.getNodeTypes()
        self.logTime()
        
        
        # nNodes=self.mesh.nodeCoords.shape[0]
        voltages=self.nodeVoltages

        # nFixedV=len(vFix2Global)
        
        # dofNodes=np.setdiff1d(range(nNodes), self.vSourceNodes)
        dof2Global=np.nonzero(self.nodeRoleTable==0)[0]
        nDoF=dof2Global.shape[0]

        # b = self.setRHS(nDoF)
        
        # M=self.getMatrix()
        
        M,b=self.getSystem()
        
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
        
    
    def analyticalEstimate(self,rvec=None):
        """
        Analytical estimate of potential field.
        
        Calculates estimated potential from sum of piecewise functions
        
              Vsrc,         r<=rSrc
        v(r)={
              isrc/(4Pi*r)
              
        If rvec is none, calculates at every node of mesh

        Parameters
        ----------
        rvec : float[:], optional
            Distances from source to evaluate at. The default is None.

        Returns
        -------
        vAna, list of float[:]
            List (per source) of estimated potentials 
        intAna, list of float
            Integral of analytical curve across specified range.

        """
        srcI=[]
        srcLocs=[]
        srcRadii=[]
        srcV=[]
  
          
        def __analytic(rad,V,I,r):
            inside=r<rad
            voltage=np.empty_like(r)
            voltage[inside]=V
            voltage[~inside]=I/(4*np.pi*r[~inside])
            
            #integral
            integral=V*rad #inside
            integral+=(I*4*np.pi)*(np.log(max(r))-np.log(rad))
            return voltage, integral
        

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
            
        vAna=[]
        intAna=[]
        for ii in nb.prange(len(srcI)):
            if rvec is None:
                r=np.linalg.norm(
                    self.mesh.nodeCoords-srcLocs[ii],
                    axis=1)
            else:
                r=rvec

            vEst,intEst=__analytic(srcRadii[ii], srcV[ii], srcI[ii], r)
            
            vAna.append(vEst)
            intAna.append(intEst)
            
        return vAna, intAna
        
    def calculateErrors(self,rvec=None):
        """
        Estimate error in solution.
        
        Estimates error between simulated solution assuming point/spherical
        sources in uniform conductivity.
        
        For the error metric to be applicable across a range of domain
        sizes and mesh densities, it must 
        
        The normalized error metric approximates the area between the 
        analytical solution i/(4*pi*sigma*r) and a linear interpolation
        between the simulated nodal voltages, evaluated across the simulation domain

        Parameters
        ----------
        rvec : float[:], optional
            Alternative points at which to evaluate the analytical solution. The default is None.
            
        Returns
        -------
        errSummary : float
            Normalized, overall error metric.
        err : float[:]
            Absolute error estimate at each node (following global node ordering)
        vAna : float[:]
            Estimated potential at each node (following global ordering)
        sorter : int[:]
            Indices to sort globally-ordered array based on the corresponding node's distance from center
            e.g. erSorted=err[sorter]
        """
        v=self.nodeVoltages

        coords=self.mesh.nodeCoords
        
        if rvec is None:
            r=np.linalg.norm(coords,axis=1)
        else:
            r=rvec
        
        vEst, intAna=self.analyticalEstimate(r)
        
        vAna=np.sum(np.array(vEst),axis=0)
        anaInt=sum(intAna)

        sorter=np.argsort(r)
        rsort=r[sorter]
        
            
        err=v-vAna
        errSort=err[sorter]
        errSummary=np.trapz(abs(errSort),rsort)/anaInt


        return errSummary, err, vAna, sorter
    
   #TODO: deprecate 
    # def setRHS(self,nDoF):
    #     """
    #     Set right-hand-side of system of equations (as determined by simulation's boundary conditions)

    #     Parameters
    #     ----------
    #     nDoF : int64
    #         Number of degrees of freedom. Equal to number of nodes with unknown voltages plus number of nodes
    #         with a fixed injected current
    #     nodeSubset : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     b : float[:]
    #         RHS of system Gv=b.

    #     """
    #     #set right-hand side
    #     self.startTiming('Setting RHS')
    #     nCurrent=len(self.currentSources)
    #     b=np.zeros(nDoF+nCurrent,dtype=np.float64)
        
    #     # for n,val in zip(self.iSourceNodes,self.iSourceVals):
    #     #     # b[global2subset(n,nDoF)]=val
    #     #     nthDoF=nodeSubset[n]
    #     #     b[nthDoF]=val
        
    #     for ii in nb.prange(nCurrent):
    #         b[nDoF+ii]=self.currentSources[ii].value
            
    #     self.logTime()
    #     self.RHS=b
        
    #     return b
    
    #TODO: deprecate
    # def getMatrix(self):
    #     """
    #     Calculates conductivity matrix G for the current mesh.

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     G: sparse matrix
    #         Conductance matrix G in system Gv=b.

    #     """
    #     self.startTiming("Filtering conductances")
    #     #diagonal elements are -sum of row
        
    #     nCurrent=len(self.currentSources)
    #     nDoF=sum(self.nodeRoleTable==0)+nCurrent
        
        
    #     edges=self.edges
    #     conductances=self.conductances
    #     b=self.RHS
        
    #     nodeA=[]
    #     nodeB=[]
    #     cond=[]
    #     for ii in nb.prange(len(conductances)):
    #         edge=edges[ii]
    #         g=conductances[ii]
            
    #         for nn in range(2):
    #             this=np.arange(2)==nn
    #             thisrole,thisDoF=self.__toDoF(edge[this][0])
    #             thatrole,thatDoF=self.__toDoF(edge[~this][0])

    #             if thisDoF is not None:
    #                 # valid degree of freedom
    #                 cond.append(g)
    #                 nodeA.append(thisDoF)
    #                 nodeB.append(thisDoF)
                    
    #                 if thatDoF is not None:
    #                     # other node is DoF
    #                     cond.append(-g)
    #                     nodeA.append(thisDoF)
    #                     nodeB.append(thatDoF)
                        
    #                 else:
    #                     #other node is fixed voltage
    #                     thatVoltage=self.nodeVoltages[edge[~this]]
    #                     b[thisDoF]-=g*thatVoltage
      
    #     self.logTime()
    
    
    #     self.startTiming("assembling system")

    #     G=scipy.sparse.coo_matrix((cond,(nodeA,nodeB)),shape=(nDoF,nDoF))
    #     G.sum_duplicates()
        
    #     self.logTime()
    #     self.gMat=G
    #     self.RHS=b
    #     return G

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
        Get an integer per node indicating its role.
        
        Type indices:
            0: Unknown voltage
            1: Fixed voltage
            2: Fixed current, unknown voltage

        Returns
        -------
        None.

        """
        self.insertSourcesInMesh()
        
        self.nDoF=sum(self.nodeRoleTable==0)
        trueDoF=np.nonzero(self.nodeRoleTable==0)[0]
        
        for n in nb.prange(len(trueDoF)):
            self.nodeRoleVals[trueDoF[n]]=n
        
    
    def getEdgeMat(self,dedup=True):
        """Return conductance matrix across all nodes in mesh.
        
        Parameters
        ----------
        dedup : bool, optional
            Sum parallel conductances. The default is True.

        Returns
        -------
        gAll : COO sparse matrix
            Conductance matrix, N x N for a mesh of N nodes.

        """
        nNodes=self.mesh.nodeCoords.shape[0]
        a=np.tile(self.conductances, 2)
        E=np.vstack((self.edges,np.fliplr(self.edges)))
        gAll=scipy.sparse.coo_matrix((a, (E[:,0], E[:,1])),
                                     shape=(nNodes,nNodes))
        if dedup:
            gAll.sum_duplicates()
            
        return gAll
    
    def getNodeConnectivity(self):
        """
        Calculate how many conductances terminate in each node.
        
        A fully-connected hex node will have 24 edges prior to merging parallel
        conductances; less than this indicates the node is hanging (nonconforming).

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        nConn : int[:]
            Number of edges that terminate in each node.

        """
        _,nConn=np.unique(self.edges.ravel(),return_counts=True)
        
        if self.mesh.nodeCoords.shape[0]!=nConn.shape[0]:
            raise ValueError('Length mismatch: %d nodes, but %d values\nIs a node unconnected?'%(
                self.mesh.nodeCoords.shape[0], nConn.shape[0]))
            
        return nConn
    
    def regularizeMesh(self):
        nConn=self.getNodeConnectivity()
        # self.__dedupEdges()
        
        badNodes=np.argwhere(nConn<24).squeeze()
        keepEdge=np.ones(self.edges.shape[0],dtype=bool)
        boundaryNodes=self.mesh.getBoundaryNodes()
        
        newEdges=[]
        newConds=[]
        
        for ii in nb.prange(badNodes.shape[0]):
            node=badNodes[ii]
            if np.isin(node,boundaryNodes):
                continue
            
            #get edges connected to hanging node
            isSharedE=np.any(self.edges==node,axis=1, keepdims=True)
            isOther=self.edges!=node
            neighbors=self.edges[isSharedE&isOther]
            #get edges connected adjacent to hanging node
            matchesNeighbor=np.isin(self.edges,neighbors)
            isLongEdge=np.all(matchesNeighbor, axis=1)
            
            #get long edges (forms triangle with edges to hanging node)
            longEdges=self.edges[isLongEdge]
            gLong=self.conductances[isLongEdge]
            

            #TODO: generalize split
            for eg,g in zip(longEdges,gLong):
                for n in eg:
                    newEdges.append([n,node])
                    newConds.append(0.5*g)
                #     shortEdge=np.array([n, node])
                #     isShort=self.__isMatchingEdge(self.edges, 
                #                                   shortEdge)
                #     theseConds=self.conductances[isShort]
                #     theseEdges=self.edges[isShort]
                    
                #     newConds.extend(theseConds.tolist())
                #     newEdges.extend(theseEdges.tolist())
                
            
            keepEdge[isLongEdge]=False
            # print('%d of %d'%(ii,badNodes.shape[0]))
            
        revisedEdges=np.vstack((self.edges[keepEdge],
                                np.array(newEdges)))
        revisedConds=np.concatenate((self.conductances[keepEdge], 
                                     np.array(newConds)))
        
        self.conductances=revisedConds
        self.edges=revisedEdges
        
        return revisedConds, revisedEdges
    
    def __dedupEdges(self):
        e=self.edges
        g=self.conductances
        nnodes=self.mesh.nodeCoords.shape[0]
        
        gdup=np.concatenate((g,g))
        edup=np.vstack((e,np.fliplr(e)))
        
        a,b=np.hsplit(edup,2)
            
        gmat=scipy.sparse.coo_matrix((gdup,(edup[:,0], edup[:,1])),
                                     shape=(nnodes,nnodes))
        gmat.sum_duplicates()
        
        tmp=scipy.sparse.tril(gmat,-1)
        
        gComp=tmp.data
        eComp=np.array(tmp.nonzero()).transpose()
        
        self.edges=eComp
        self.conductances=gComp
            
            
    def __isMatchingEdge(self,edges,toMatch):
        nodeMatches=np.isin(edges,toMatch)
        matchingEdge=np.all(nodeMatches, axis=1)
        return matchingEdge
            
    
    def getSystem(self):
        """
        Construct system of equations GV=b.
        
        Rows represent each node without a voltage or current 
        constraint, followed by an additional row per current
        source.

        Returns
        -------
        G : COO sparse matrix
            Conductances between degrees of freedom.
        b : float[:]
            Right-hand side of system, representing injected current
            and contributions of fixed-voltage nodes.

        """
        # global mat is NxN
        # for Ns current sources, Nf fixed nodes, and Nx floating nodes, 
        # N - nf = Ns + Nx =Nd
        # system is Nd x Nd
        
        isSrc=self.nodeRoleTable==2
        isFix=self.nodeRoleTable==1
        
        # N=self.mesh.nodeCoords.shape[0]
        Ns=len(self.currentSources)
        Nx=np.nonzero(self.nodeRoleTable==0)[0].shape[0]
        Nd=Nx+Ns
        Nf=np.nonzero(isFix)[0].shape[0]
        N_ext=Nd+Nf
        
        self.startTiming("Filtering conductances")
        #renumber nodes in order of dof, current source, fixed v
        dofNumbering=self.nodeRoleVals.copy()
        dofNumbering[isSrc]=Nx+dofNumbering[isSrc]
        dofNumbering[isFix]=Nd+np.arange(Nf)
        
        edges=dofNumbering[self.edges]
        
        #filter bad vals (both same DoF)
        isValid=edges[:,0]!=edges[:,1]
        evalid=edges[isValid]
        cvalid=self.conductances[isValid]
        
        # duplicate for symmetric matrix
        Edup=np.vstack((evalid, np.fliplr(evalid)))
        cdup=np.tile(cvalid,2)

        
        #get only DOF rows/col
        isRowDoF=Edup[:,0]<Nd
        
        #Fill matrix with initial degrees of freedom
        E=Edup[isRowDoF]
        c=cdup[isRowDoF]
        
        self.logTime()
        self.startTiming("assembling system")
        G=scipy.sparse.coo_matrix(
            (-c, (E[:,0], E[:,1]) ),
                                    shape=(Nd,N_ext))

        gR=G.tocsr()
        
        v=np.zeros(N_ext)
        v[Nd:]=self.nodeVoltages[isFix]

        b=np.array(gR.dot(v)).squeeze()
        
        for ii in range(Ns):
            b[ii+Nx]+=self.currentSources[ii].value
        
        diags=-np.array(gR.sum(1)).squeeze()

        G.setdiag(diags)
        G.resize(Nd,Nd)
        
        self.logTime()
        
        self.RHS=b
        self.gMat=G
        return G, b
        
     
    
    def selGlobalByDoF(self,nthDoF):
        nDoF=sum(self.nodeRoleTable==0)
        
        if nthDoF>=nDoF:
            #select current source
            nval=(self.nodeRoleTable==2)
        else:
            nval=(self.nodeRoleTable==0)
            
        bSel=nval&(self.nodeRoleVals==nthDoF)
        return bSel
                

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
        Get element containing specified point.

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
        Insert element into the mesh.

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
        Get the discrete conductances from every element.

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
            #TODO: remove deprecated
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
            #TODO: deprecated function
            gnodes=util.sparse2denseIndex(onodes, self.indexMap)
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
            xyz=util.index2pos(nn,nX)
            if np.any(xyz==0) or np.any(xyz==(nX-1)):
                bnodes.append(ii)
        
        return np.array(bnodes)




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
            xyz+=step*util.toBitArray(idx)
            
        ownstep=2**(maxdepth-self.depth)
        
        ownXYZ=[xyz+ownstep*util.toBitArray(i) for i in range(8)]
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
            offset=util.toBitArray(ii)*newSpan
            newOrigin=self.origin+offset
            newIndex=self.index.copy()
            newIndex.append(ii)
            self.children.append(Octant(newOrigin,newSpan,self.depth+1,index=newIndex))
            
        # return self.children
    def getOwnCoords(self):
        return [self.origin+self.span*util.toBitArray(n) for n in range(8)]

    
    def getCoordsRecursively(self,bbox,maxdepth):
        # T=Logger('depth  %d'%self.depth, printStart=False)
        if len(self.children)==0:
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
        basepath=os.path.join(self.studyPath)
        
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