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
import time
import os
import resource 
import pickle


nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0

# @nb.njit(int64[:](int64,int64))
# @nb.njit
# def toBitArray(val):
#     # if bits==0:
#     #     bits=np.ceil(np.log2(val))
    
#     return np.array([(val>>ii)&1 for ii in range(3)])
@nb.njit
def uniformResample(origin,span,nPts):
    vx,vy,vz=[np.linspace(o,o+s,nPts) for o,s in zip(origin,span)]
    
    XX,YY,ZZ=np.meshgrid(vx,vy,vz)


    coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()

    
    return coords

# @nb.njit
def getElementInterpolant(element,nodeVals):    
    coords=element.getCoordsRecursively()
    coefs=np.array([coord2InterpVals(xyz) for xyz in coords])

        
    interpCoefs=np.linalg.solve(coefs, nodeVals)
    return interpCoefs
    
@nb.njit
def evalulateInterpolant(interp,location):
    
    # coeffs of a, bx, cy, dz, exy, fxz, gyz, hxyz
    varList=coord2InterpVals(location)
    
    # interpVal=np.matmul(interp,varList)
    interpVal=np.dot(interp,varList)
    
    return interpVal

@nb.njit
def coord2InterpVals(coord):
    x,y,z=coord
    return np.array([1,
                     x,
                     y,
                     z,
                     x*y,
                     x*z,
                     y*z,
                     x*y*z]).transpose()
@nb.njit
def getCurrentVector(interpolant,location):
    #coeffs are 
    #0  1   2   3    4    5    6    7
    #a, bx, cy, dz, exy, fxz, gyz, hxyz
    #gradient is [
    #   [b + ey + fz + hyz],
    #   [c + ex + gz + hxz],
    #   [d + fx + gy + hxy]
    
    varList=coord2InterpVals(location)
    
    varSets=np.array([[0,2,3,6],
                      [0,1,3,5],
                      [0,1,2,4]])
    coefSets=np.array([[1,4,5,7],
                       [2,4,6,7],
                       [3,5,6,7]])
    
    varVals=np.array([varList[n] for n in varSets])
    coefVals=np.array([interpolant[n] for n in coefSets])
    
    vecVals=np.array([-np.dot(v,c) for v,c in zip(varVals,coefVals)])

    return vecVals


@nb.njit()
# @nb.njit(['int64[:](int64, int64)', 'int64[:](int64, Omitted(int64))'])
def toBitArray(val,nBits=3):
    return np.array([(val>>n)&1 for n in range(nBits)])

@nb.njit
def anyMatch(searchArray,searchVals):
    """
    Rapid search if any matches occur (returns immediately at first match)

    Parameters
    ----------
    searchArray : array
        Array to seach.
    searchVals : array
        Values to search array for.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    for el in searchArray.ravel():
        if any(np.isin(searchVals,el)):
            return True
    
    return False
    

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
    def __init__(self,resultPath,extents=np.zeros(3)):
        self.iSourceCoords=[]
        self.iSourceNodes=[]
        self.iSourceVals=[]
        self.vSourceCoords=[]
        self.vSourceVals=[]
        self.vSourceNodes=[]
        
        self.mesh=Mesh(extents)
        self.currentTime=0.
        
        self.stepLogs=[]
        self.stepTime=[]
        self.memUsage=0
        
        self.nodeVoltages=np.empty(0)
        self.edges=[[]]
        self.gMat=[]
        self.RHS=[]
        
        self.resultPath=resultPath
        self.meshtype='uniform'
        
        self.ptPerAxis=0
        
        self.iteration=0
        
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
            engFormat=mpl.ticker.EngFormatter(unit='b')
            print(engFormat(mem*1024)+" used")
        
        return mem
    
    def makeTableHeader(self):
        cols=[
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
        
        
#TODO: account for non-nodal sources   
    def addCurrentSource(self,value,coords=None,index=None):
        """
        Attaches a current source to the simulation.
        Currently assumes source is located at a node

        Parameters
        ----------
        value : float
            Magnitude of current in amperes.
        coords : float[:], optional
            Cartesian coordinates of source. The default is None.
        index : int, optional
            Nodal index of source. The default is None.

        Returns
        -------o
        None.

        """
        if index is not None:
            self.iSourceNodes.append(index)
            self.iSourceVals.append(value)
            
        else:
            self.mesh.nodeCoords.append(coords)

#TODO: account for non-nodal sources        
    def addVoltageSource(self,value,coords=None,index=None):
        """
        Add voltage source to simulation.
        Currently assumes source is located at a node

        Parameters
        ----------
        value : float
            Source potential in volts.
        coords : float[:], optional
            Cartesian coordinates of source. The default is None.
        index : int, optional
            Nodal index of source. The default is None.

        Returns
        -------
        None.

        """
        if index is not None:
            coords=self.mesh.nodeCoords[index]
            self.vSourceNodes.append(index)
            self.vSourceVals.append(value)
        
    
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

        nFixedV=len(vFix2Global)
        voltages[self.vSourceNodes]=self.vSourceVals
        
        # dofNodes=np.setdiff1d(range(nNodes), self.vSourceNodes)
        nDoF=len(dof2Global)
        #TODO: handle non-nodal sources
        
        b = self.setRHS(nDoF,nodeSubset)
        
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
        nodeType,nodeSubset, vFix2Global, iFix2Global, dof2Global = self.getNodeTypes()
        self.logTime()
        
        
        nNodes=self.mesh.nodeCoords.shape[0]
        voltages=np.empty(nNodes,dtype=np.float64)

        nFixedV=len(vFix2Global)
        voltages[self.vSourceNodes]=self.vSourceVals
        
        # dofNodes=np.setdiff1d(range(nNodes), self.vSourceNodes)
        nDoF=len(dof2Global)
        #TODO: handle non-nodal sources
        
        b = self.setRHS(nDoF,nodeSubset)
        
        M=self.getMatrix(nNodes,edges,conductances,nodeType,nodeSubset,b,dof2Global)
        
        self.startTiming('Solving')
        vDoF,_=cg(M.tocsc(),b,vGuess,tol)
        self.logTime()
        
        voltages[dof2Global]=vDoF
        
        self.nodeVoltages=voltages
        return voltages
    
    def calculateErrors(self,srcAmplitude,srcType,nX=100,showPlots=False,savePlots=False):
        
        nTypes,_,_,_,_=self.getNodeTypes()
        v=self.nodeVoltages
        rest=nTypes==0
        
        coords=self.mesh.nodeCoords
        edges=self.mesh.edges
        
        nNodes=coords.shape[0]
        nElems=len(self.mesh.elements)
        
        r=np.linalg.norm(self.mesh.nodeCoords,axis=1)
        rDense=np.linspace(0,max(r[rest]),100)
        
        sorter=np.argsort(r)
        rsort=r[sorter]
        vsort=v[sorter]
        
        
        
        
        #TODO: extend to multiple sources
        analytic=analyticVsrc(np.zeros(3), srcAmplitude, rsort,srcType=srcType)
        analyticDense=analyticVsrc(np.zeros(3), srcAmplitude, rDense,srcType=srcType)
        
        err=vsort-analytic
        # FVU,err=getFVU(v, analytic, rest)
   
        # errAll=np.zeros(coords.shape[0])
        # errAll[rest]=err
        FVU=np.trapz(abs(err),rsort)/np.trapz(analytic,rsort)

        # if showPlots:
            
        #     figResult=plt.figure()
        #     showSlice(coords,v, nX,plotWhich=None,edges=edges)
        #     plt.title('Simulated solution [V]')
        #     plt.tight_layout()
            
        
        #     figImage=plt.figure()
        #     showSlice(coords,errAll, nX,plotWhich=None,edges=edges,forceBipolar=True)
        #     plt.title('Absolute error [V]')
        #     plt.tight_layout()
            
        #     fig2d, axes=plt.subplots(2,1)
        #     ax2dA,ax2dB=axes
        #     ax2dA.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        #     ax2dA.plot(rDense,analyticDense, label='Analytical')
        #     ax2dA.scatter(r[rest],v[rest],c='r',label='Simulation')
        #     ax2dA.legend()
        #     ax2dA.set_title('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
        #     ax2dA.set_xlabel('Distance from source [m]')
        #     ax2dA.set_ylabel('Voltage [V]')
            
        #     ax2dB.scatter(r[rest],err,c='r',label='Absolute')
        #     ax2dB.set_ylabel('Absolute error [V]')
        #     ax2dB.sharex(ax2dA)
        #     plt.tight_layout()
            
        #     if savePlots:
        #         figResult.savefig(self.resultPath+'result_'+self.iteration)
        #         figImage.savefig(self.resultPath+'errorImage_'+self.iteration)
        #         fig2d.savefig(self.resultPath+'errorPlot_'+self.iteration)
        
        return FVU
    
    
    def setRHS(self,nDoF,nodeSubset):
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
        b=np.zeros(nDoF,dtype=np.float64)
        
        for n,val in zip(self.iSourceNodes,self.iSourceVals):
            # b[global2subset(n,nDoF)]=val
            nthDoF=nodeSubset[n]
            b[nthDoF]=val
            
        self.logTime()
        self.RHS=b
        
        return b
    
    
    def getMatrix(self,nNodes,edges,conductances,nodeType,nodeSubset, b,dof2Global):
        """
        Calculates conductivity matrix G for the current mesh.

        Parameters
        ----------
        nNodes : int
            Total number of nodes.
        edges : [[]]
            Pairs of node indices indicating conductive mesh edge.
        conductances : float[:]
            Conductance [in s] of edge.
        nodeType : int[:]
            Indicates node characteristics (0=unknown, 1=fixed voltage, 2= fixed current).
        nodeSubset : TYPE
            DESCRIPTION.
        b : float[:]
            RHS of equations.
        dof2Global : int[:]
            Array mapping degree of freedom numbering to global node numbering.

        Returns
        -------
        G: sparse matrix
            Conductance matrix G in system Gv=b.

        """
        #TODO: parallelize it
        self.startTiming("Filtering conductances")
        #diagonal elements are -sum of row
        
        nNodes=self.mesh.nodeCoords.shape[0]
        nDoF=len(dof2Global)

        diags=np.zeros(nNodes,dtype=np.float64)
        
        gDoF=[]
        gDofNodes=[]
        for edge,g in zip(edges,conductances):     
            #adjust diagonals
            diags[edge[0]]+=g
            diags[edge[1]]+=g
            # isFixed=np.isin(edge,self.vSourceNodes)
            isFixed=nodeType[edge]==1
            numfixed=np.sum(isFixed)
            # only adjust RHS if one node is fixed V
            if numfixed==2:
                # both nodes fixed; ignore
                continue
            else: 
                if numfixed==1:
                    # one node fixed; adjust RHS for other node
                    
                    nVGlobal=edge[isFixed]
                    nFreeGlobal=edge[~isFixed]
                    
                    nthV=nodeSubset[nVGlobal]
                    nthDoF=nodeSubset[nFreeGlobal]
                    
                    
                    # nthV=global2subset(nVGlobal, self.vSourceNodes)
                    # nthDoF=global2subset(nFreeGlobal, dofNodes)
                    
                    # b[nthDoF]-=self.vSourceVals[nthV.ravel()[0]]*g
                    b[nthDoF]+=self.vSourceVals[nthV[0]]*g
                    
                else:
                    # both nodes are DoF
                    gDoF.append(-g)
                    dofEdge=np.array([nodeSubset[e] for e in edge])
                    gDofNodes.append(dofEdge)
        
        
        diagVals=diags[dof2Global]
        
        self.logTime()
        
        gDofNodes=np.array(gDofNodes)
        nodeA=gDofNodes[:,0]
        nodeB=gDofNodes[:,1]  
        diagIndex=np.arange(nDoF)
        
        nA=np.concatenate((nodeA, nodeB,diagIndex))
        nB=np.concatenate((nodeB,nodeA,diagIndex))
        cond2=np.concatenate((gDoF,gDoF,diagVals))       
    
        self.startTiming("assembling system")
        # double up for matrix symmetry

        
        G = scipy.sparse.coo_matrix((cond2, (nA,nB)), shape=(nDoF, nDoF))
        G.sum_duplicates()
        
        self.logTime()
        self.gMat=G
        return G

    
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
        nNodes=self.mesh.nodeCoords.shape[0]        
        nFixedV=len(self.vSourceNodes)
        nFixedI=len(self.iSourceNodes)
        
        nDoF=nNodes-nFixedV+nFixedI
        
        dof2Global=[]
        vFix2Global=[]
        iFix2Global=[]
        
        nodeType=np.zeros(nNodes,dtype=np.int64)
        global2Subset=np.empty(nNodes,dtype=np.int64)

        nodeType[self.vSourceNodes]=1
        nodeType[self.iSourceNodes]=2

        for sub,n in enumerate(self.vSourceNodes):
            global2Subset[n]=sub

        
        for n in nb.prange(nNodes):
            typ=nodeType[n]
            if typ==0:
                global2Subset[n]=len(dof2Global)
                dof2Global.append(n)
        
        #add current sources to list of DoFs
        nFree=len(dof2Global)
        dof2Global.extend(self.iSourceNodes)
        for nthI,glob in enumerate(self.iSourceNodes):
            global2Subset[glob]=nFree+nthI
        
        return nodeType, global2Subset, np.array(vFix2Global), np.array(iFix2Global), np.array(dof2Global)
        
    
    def resamplePlane(self,nXSamples=None,normalAxis=2,axisLocation=0.):
        
        xmax=max(self.mesh.extents)/2

        if nXSamples is None:
            nXSamples=self.ptPerAxis
                    
        coords=self.mesh.nodeCoords
        sel=np.equal(coords[:,normalAxis],axisLocation)
        selAx=np.array([n for n in range(3) if n!=normalAxis])
            
        planeCoords=coords[:,selAx]
        sliceCoords=planeCoords[sel]
        
        xx=np.linspace(-xmax,xmax,nXSamples)
        XX,YY=np.meshgrid(xx,xx)
        
        interpCoords=np.vstack((XX.ravel(),YY.ravel(),np.zeros_like(XX.ravel()))).transpose()
        vInterp=np.empty_like(XX.ravel())
        
        for ii in range(len(interpCoords)):
            interpCoord=interpCoords[ii]
            container=self.mesh.getContainingElement(interpCoord)
            contNodes=container.globalNodeIndices
            contV=self.nodeVoltages[contNodes]
            interp=getElementInterpolant(container, contV)
            vInterp[ii]=evalulateInterpolant(interp, interpCoord)
            
        return vInterp.reshape((nXSamples,nXSamples)), interpCoords
    
def getMappingArrays(generalArray,subsetArray):
    gen2sub=-np.ones(len(generalArray))
    sub2gen=[np.argwhere(n==generalArray).squeeze() for n in subsetArray]
    
    return gen2sub, sub2gen
        
class Mesh:
    def __init__(self,extents,elementType='Admittance'):
        self.extents=extents
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
            
       
    def addElement(self,origin, extents,sigma,nodeIndices):
        if self.elementType=='Admittance':
            newEl=AdmittanceHex(origin,extents,sigma)
        elif self.elementType=='FEM':
            newEl=FEMHex(origin,extents,sigma)
            
        newEl.setGlobalIndices(nodeIndices)
        self.elements.append(newEl)
        
    def getConductances(self):
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
        l0Min=np.infty
        
        for el in self.elements:
            l0Min=min(l0Min,el.l0)
        return l0Min

class Logger():
    def __init__(self,stepName):
        self.name=stepName
        print(stepName+" starting")
        self.start=time.process_time()
        self.duration=0
        self.memory=0

        
    def logCompletion(self):
        tend=time.process_time()
        duration=tend-self.start
        engFormat=mpl.ticker.EngFormatter()
        print(self.name+": "+engFormat(duration)+ " seconds")
        self.duration=duration       
        self.memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
@nb.njit
def distMetric(evalLocation,srcLocation,iVal,sigma):
    delta=evalLocation-srcLocation
    dist=np.linalg.norm(delta/sigma)
    return iVal/(4*np.pi*dist)
        

class Octree():
    def __init__(self,boundingBox,maxDepth=10):
        self.center=np.mean(boundingBox.reshape(2,3),axis=0)
        self.span=(boundingBox[3:]-boundingBox[:3])
        self.maxDepth=maxDepth
        self.bbox=boundingBox
        self.indexMap=np.empty(0,dtype=np.int64)
        
        coord0=self.center-self.span/2
        
        self.tree=Octant(coord0,self.span)
        
        
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
            idx=sparse2denseIndex(idx,self.indexMap)
        return idx
        
    def makeMesh(self,mesh):
        octs=self.tree.getTerminalOctants()
        coords=self.getCoordsRecursively()
        mesh.nodeCoords=coords
        
        for o in octs:
            # ocoords=o.getOwnCoords()
            mesh.addElement(o.origin,o.span,np.ones(3),o.globalNodes)
        
        return mesh
    
    def printStructure(self):
        self.tree.printStructure()
        
    def octantByList(self,indexList,octant=None):
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
    
    #TODO: desperately slow. need to fix
    def getCoordsRecursively(self):
        coords,i=self.tree.getCoordsRecursively(self.bbox,self.maxDepth)
        indices=np.array(i)
        self.indexMap=indices
        
        for o in self.tree.getTerminalOctants():
            tstNodes=o.globalNodes
            # newNodes=[np.argwhere(indices==n).squeeze() for n in tstNodes]
            newNodes=[sparse2denseIndex(n,np.array(indices)) for n in tstNodes]
            o.globalNodes=np.array(newNodes)
        return np.array(coords)
    

  
@nb.njit()
def sparse2denseIndex(sparseVal,denseList):
    
    for n,val in np.ndenumerate(denseList):
        if val==sparseVal:
            return n[0]
        
    # return None
        
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
    def __init__(self,origin, span,depth=0,index=0):
        self.origin=origin
        self.span=span
        self.center=origin+span/2
        self.l0=np.prod(span)**(1/3)
        self.children=[]
        self.depth=depth
        self.globalNodes=np.empty(8,dtype=np.int64)
        self.nX=2
        self.index=index
        self.nodeIndices=-np.ones(27,dtype=np.int64)
        

        
    def calcGlobalIndices(self,globalBbox,maxdepth):
        x0=globalBbox[:3]
        nX=2**maxdepth
        dX=(globalBbox[3:]-x0)/(nX)
        coords=self.getOwnCoords()
        
        ndxOffsets=np.array([(nX+1)**n for n in range(3)])
        
        for N,c in enumerate(coords):
            idxArray=(c-x0)/dX
            ndx=np.dot(ndxOffsets,idxArray)
            self.globalNodes[N]=ndx
            
        return self.globalNodes
            
        
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
            self.children.append(Octant(newOrigin,newSpan,self.depth+1,ii))
            
        # return self.children
    def getOwnCoords(self):
        return [self.origin+self.span*toBitArray(n) for n in range(8)]
    
    
    def getCoordsRecursively(self,bbox,maxdepth):
        if len(self.children)==0:
            # coords=[self.origin+self.span*toBitArray(n) for n in range(8)]
            coords=self.getOwnCoords()
            # indices=self.globalNodes
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
            
            self.globalNodes=indices
            
            coords=np.array(coordList)[sel]
            coords=coords.tolist()
                
        return coords, indices.tolist()
    
    def index2pos(self,ndx,dX):
        arr=[]
        for ii in range(3):
            arr.append(ndx%dX)
            ndx=ndx//dX
        return np.array(arr)
    
    def pos2index(self,pos,dX):
        vals=np.array([dX**n for n in range(3)])
        newNdx=np.dot(vals,pos)
        return np.rint(newNdx)
    
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
            
        sim=Simulation(simName,extents=self.span)
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
        
        fname=os.path.join(self.studyPath,simulation.resultPath+'.p')
        pickle.dump(simulation,open(fname,'wb'))
        
    def loadData(self,simName):
        fname=os.path.join(self.studyPath,simName+'.p')
        data=pickle.load( open(fname,'rb'))

        return data
        
    def savePlot(self,fig,fileName,ext):
        basepath=os.path.join(self.studyPath,self.currentSim.resultPath)
        
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fname=os.path.join(basepath,fileName+ext)
        plt.savefig(fname)
        
    def animatePlot(self,plotfun,aniName=None):
        ims=[]
        cmax=0.
        cmin=0.
        files=os.listdir(self.studyPath)
        
        fnames=[]
        
        for n in files:
            name,ext=os.path.splitext(n)
            if ext=='.p':
                fnames.append(name)
            
        fnames.sort()
        
        fig=plt.figure()
        for ii,name in enumerate(fnames):
            dat=self.loadData(name)
            im=plotfun(fig,dat)
            txt=fig.text(0.01,0.95,'frame %d'%ii,
                           horizontalalignment='left',verticalalignment='bottom')
            im.append(txt)
            ims.append(im)
        
        # def aniFun(ii):
        #     sim=self.loadData(fnames[ii])
        #     art=plotfun(fig,sim)
        #     txt=fig.text(1,1,'frame %d'%ii,
        #                  horizontalalignment='left',verticalalignment='bottom')
        #     art.append(txt)
        #     return art
            
            
        ani=mpl.animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=2000,blit=False)
        # ani=mpl.animation.FuncAnimation(fig, aniFun, interval=1000,repeat_delay=2000,blit=True)
        if aniName is None:
            plt.show()
        else:
            ani.save(os.path.join(self.studyPath,aniName+'.mp4'),fps=1)
            
        return ani