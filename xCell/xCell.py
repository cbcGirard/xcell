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
from scipy.sparse.linalg import spsolve
from Visualizers import *
import time


nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0

# @nb.njit(int64[:](int64,int64))
# @nb.njit
# def toBitArray(val):
#     # if bits==0:
#     #     bits=np.ceil(np.log2(val))
    
#     return np.array([(val>>ii)&1 for ii in range(3)])

@nb.njit
def toBitArray(val,nBits=3):
    return np.array([(val>>n)&1 for n in range(nBits)])

@nb.njit
def anyMatch(searchArray,searchVals):
    for el in searchArray.ravel():
        if any(np.isin(searchVals,el)):
            return True
    
    return False
    

@nb.experimental.jitclass([
    ('origin', float64[:]),
    ('extents',float64[:]),
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class Element:
    def __init__(self,origin, extents,sigma):
        self.origin=origin
        self.extents=extents
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoords(self):
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
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class FEMHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoords(self):
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
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class AdmittanceHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
        self.sigma=sigma
        self.globalNodeIndices=np.empty(8,dtype=np.int64)
        
    def getCoords(self):
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
    def __init__(self):
        self.iSourceCoords=[]
        self.iSourceNodes=[]
        self.iSourceVals=[]
        self.vSourceCoords=[]
        self.vSourceVals=[]
        self.vSourceNodes=[]
        
        self.mesh=Mesh(np.array([0,0,0]))
        self.currentTime=0.
        
        self.stepLogs=[]
        self.stepTime=[]
        
    def startTiming(self,stepName):
        self.stepLogs.append(Logger(stepName))
        
    def logTime(self):
        self.stepLogs[-1].logCompletion()
        
        
    # def logParams(self,fileName):
    #     f=open(fileName,'a')
    #     f.write('#Mesh params')
    #     f.write('Bounds, '+','.join(map(str,self.mesh.extents)))
    #     meshCats=["Element Type","Elements","Nodes"]
    #     meshVals=[self.mesh.elementType,len(self.mesh.elements),]
    #     f.write('Element Type, %s\nElements, %d\nNodes, %d'% self.mesh.elementType)
        
    #     f.write()
        
    #     f.write("#Step Times")
    #     tAll=0
    #     for name,dt in zip(self.stepName,self.stepTime):
    #         f.write('%s, %g\n'%(name,dt))
    #         tAll+=dt
    #     f.write('Total,%g\n'%tAll)
    #     f.write('## Step Times')
        
    #     f.close()
    
    def makeTableHeader(self):
        cols=[
            "Element type",
            "Number of nodes",
            "Number of elements",
            "RMS error",
            "MakeElements",
            "Calc Cunductances",
            "Sort Nodes",
            "Set RHS",
            "Filter conductances",
            "Assemble system",
            "Solve system",
            "Total time"]
        return ','.join(cols)
    
    def logAsTableEntry(self,csvFile,RMSerror):
        f=open(csvFile,'a')
        data=[
            self.mesh.elementType,
            self.mesh.nodeCoords.shape[0],
            len(self.mesh.elements),
            RMSerror
            ]
        dt=0
        for log in self.stepLogs:
            dt+=log.duration
            data.append(log.duration)
            
        data.append(dt)
        
        f.write(','.join(map(str,data))+'\n')
        f.close()
        
        
        
    def addCurrentSource(self,value,coords=None,index=None):
        if index is not None:
            self.iSourceNodes.append(index)
            self.iSourceVals.append(value)
            
    def addVoltageSource(self,value,coords=None,index=None):
        if index is not None:
            self.vSourceNodes.append(index)
            self.vSourceVals.append(value)
        
    
    def solve(self):
        self.startTiming("calculate conductances")
        edges,conductances=self.mesh.getConductances()
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

        
        
        #set right-hand side
        self.startTiming('Setting RHS')
        b=np.zeros(nDoF,dtype=np.float64)
        
        gDoF=[]
        gDofNodes=[]
        
        for n,val in zip(self.iSourceNodes,self.iSourceVals):
            # b[global2subset(n,nDoF)]=val
            nthDoF=nodeSubset[n]
            b[nthDoF]=val
            
        self.logTime()
        
        #TODO: parallelize it
        self.startTiming("Filtering conductances")
        #diagonal elements are -sum of row
        diags=np.zeros(nNodes,dtype=np.float64)
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
                    b[nthDoF]-=self.vSourceVals[nthV[0]]*g
                    
                else:
                    # both nodes are DoF
                    gDoF.append(-g)
                    dofEdge=np.array([nodeSubset[e] for e in edge])
                    gDofNodes.append(dofEdge)
        
        diagIndex=np.arange(nDoF)
        diagVals=diags[dof2Global]
        
        self.logTime()
        
        gDofNodes=np.array(gDofNodes)
        
        nodeA=gDofNodes[:,0]
        nodeB=gDofNodes[:,1]  
        

        self.startTiming("assembling system")
        # double up for matrix symmetry
        nA=np.concatenate((nodeA, nodeB,diagIndex))
        nB=np.concatenate((nodeB,nodeA,diagIndex))
        cond2=np.concatenate((gDoF,gDoF,diagVals))
        
        M = scipy.sparse.coo_matrix((cond2, (nA,nB)), shape=(nDoF, nDoF))
        M.sum_duplicates()
        
        self.logTime()
        
        # compress

        self.startTiming('Solving')
        vDoF=spsolve(M.tocsc(), b)
        self.logTime()
        
        voltages[dof2Global]=vDoF
        
        return voltages
    
    def getNodeTypes(self):
        nNodes=self.mesh.nodeCoords.shape[0]        
        nFixedV=len(self.vSourceNodes)
        nFixedI=len(self.iSourceNodes)
        
        dof2Global=[]
        vFix2Global=[]
        iFix2Global=[]
        
        nodeType=np.empty(nNodes,dtype=np.int64)
        global2Subset=np.empty(nNodes,dtype=np.int64)

        
        for n in nb.prange(nNodes):
            if np.isin(n,self.vSourceNodes):
                global2Subset[n]=len(vFix2Global)
                vFix2Global.append(n)
                nodeType[n]=1
                
            elif np.isin(n,self.iSourceNodes):
                    global2Subset[n]=len(iFix2Global)
                    iFix2Global.append(n)
                    nodeType[n]=2
                
            else:
                global2Subset[n]=len(dof2Global)
                dof2Global.append(n)
                nodeType[n]=0
                
        nFree=len(dof2Global)
        dof2Global.extend(iFix2Global)
        for n in self.iSourceNodes:
            global2Subset[n]+=nFree
        
        return np.array(nodeType), np.array(global2Subset), np.array(vFix2Global), np.array(iFix2Global), np.array(dof2Global)
        
    
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
        
        
    def getContainingElement(coords):
        nElements=len(elements)
        
        for nn in nb.prange(nElem):
            elem=self.elements[nn]
            delta=coords-elem.origin
            difs=np.logical_and(delta>0,delta<elem.extents)
            if all(difs):
                return nn
           
        raise ValueError('Point (%s) not inside any element' % ' '.join(map(coords)))
            
       
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
        
        return edgeIndices,conductances
    


class Logger():
    def __init__(self,stepName):
        self.name=stepName
        print(stepName+" starting")
        self.start=time.process_time()
        self.duration=0

        
    def logCompletion(self):
        tend=time.process_time()
        duration=tend-self.start
        engFormat=mpl.ticker.EngFormatter()
        print(self.name+": "+engFormat(duration)+ " seconds")
        self.duration=duration            
        
@nb.njit
def distMetric(evalLocation,srcLocation,iVal,sigma):
    delta=evalLocation-srcLocation
    dist=np.linalg.norm(delta/sigma)
    return iVal/(4*np.pi*dist)
        
