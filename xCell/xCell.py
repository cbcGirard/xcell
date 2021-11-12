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
from os.path import exists
import resource 


nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0

# @nb.njit(int64[:](int64,int64))
# @nb.njit
# def toBitArray(val):
#     # if bits==0:
#     #     bits=np.ceil(np.log2(val))
    
#     return np.array([(val>>ii)&1 for ii in range(3)])

def uniformResample(origin,span,nPts):
    vx,vy,vz=[np.linspace(o,o+s,nPts) for o,s in zip(origin,span)]
    
    XX,YY,ZZ=np.meshgrid(vx,vy,vz)


    coords=np.vstack((XX.ravel(),YY.ravel(), ZZ.ravel())).transpose()

    
    return coords

def getElementInterpolant(element,nodeVals):    
    coords=element.getCoordsRecursively()
    coefs=np.array([coord2InterpVals(xyz) for xyz in coords])

        
    interpCoefs=np.linalg.solve(coefs, nodeVals)
    return interpCoefs
    
def evalulateInterpolant(interp,location):
    
    # coeffs of a, bx, cy, dz, exy, fxz, gyz, hxyz
    varList=coord2InterpVals(location)
    
    interpVal=np.matmul(interp,varList)
    
    return interpVal
    
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
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class FEMHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
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
    ('sigma',float64[:]),
    ('globalNodeIndices',int64[:])
    ])
class AdmittanceHex():
    def __init__(self, origin, extents, sigma):
        self.origin=origin
        self.extents=extents
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
            "Domain size",
            "Element type",
            "Number of nodes",
            "Number of elements",
            "RMS error",
            "MakeElements",
            "Calc Conductances",
            "Sort Nodes",
            "Set RHS",
            "Filter conductances",
            "Assemble system",
            "Solve system",
            "Total time",
            "Max memory"]
        return ','.join(cols)
    
    def logAsTableEntry(self,csvFile,RMSerror,extraCols=None, extraVals=None):
        oldfile=exists(csvFile)
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
            RMSerror
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
        
        
        
    def addCurrentSource(self,value,coords=None,index=None):
        if index is not None:
            self.iSourceNodes.append(index)
            self.iSourceVals.append(value)
            
    def addVoltageSource(self,value,coords=None,index=None):
        if index is not None:
            self.vSourceNodes.append(index)
            self.vSourceVals.append(value)
        
    
    def solve(self):
        self.startTiming("Calculate conductances")
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
        
        b = self.setRHS(nDoF,nodeSubset)
        
        M=self.getMatrix(nNodes,edges,conductances,nodeType,nodeSubset,b,dof2Global)
        
        self.startTiming('Solving')
        vDoF=spsolve(M.tocsc(), b)
        self.logTime()
        
        voltages[dof2Global]=vDoF
        
        return voltages


    def iterativeSolve(self,vGuess,tol=1e-5):
        self.startTiming("Calculate conductances")
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
        
        b = self.setRHS(nDoF,nodeSubset)
        
        M=self.getMatrix(nNodes,edges,conductances,nodeType,nodeSubset,b,dof2Global)
        
        self.startTiming('Solving')
        vDoF,_=cg(M.tocsc(),b,vGuess,tol)
        self.logTime()
        
        voltages[dof2Global]=vDoF
        
        return voltages
        
        
    def setRHS(self,nDoF,nodeSubset):
        #set right-hand side
        self.startTiming('Setting RHS')
        b=np.zeros(nDoF,dtype=np.float64)
        
        for n,val in zip(self.iSourceNodes,self.iSourceVals):
            # b[global2subset(n,nDoF)]=val
            nthDoF=nodeSubset[n]
            b[nthDoF]=val
            
        self.logTime()
        
        return b
    
    
    def getMatrix(self,nNodes,edges,conductances,nodeType,nodeSubset, b,dof2Global):
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

        
        M = scipy.sparse.coo_matrix((cond2, (nA,nB)), shape=(nDoF, nDoF))
        M.sum_duplicates()
        
        self.logTime()
        return M

    
    def getNodeTypes(self):
        
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
            
        # for sub,n in enumerate(self.iSourceNodes):
        #     global2Subset[n]=sub
            
        
        
        
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
        
        self.surfaceNodes=[]
        self.innerNodes=[]
        
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
        
    def makeChildren(self,division=np.array([0.5,0.5,0.5])):
        newSpan=self.span*division
        
        
        for ii in range(8):
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
    
    def refineByMetric(self,l0Function,maxDepth):
        l0Target=l0Function(self.center)
        # print("target\t%g"%l0Target)
        # print("l0\t\t%g"%self.l0)
        # print(self.center)
        if (self.l0>l0Target) and (self.depth<maxDepth):
            # print('\n\n')
            # print('depth '+str(self.depth)+', child'+str(self.index))

            self.makeChildren()
            for ii in range(8):
                self.children[ii].refineByMetric(l0Function,maxDepth)
                
    def printStructure(self):
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

    
        
    # def getConductances(self,sigma):
    #     adm=AdmittanceHex(self.origin,self.span,sigma)
    #     adm.setGlobalIndices(self.globalNodes)
    #     conds=adm.getConductanceVals()
    #     inds=adm.getConductanceIndices()
    #     # globalInds=[self.globalNodes[l] for l in localInds]
        
    #     for ch in self.children:
    #         chConds,chInds=ch.getConductances(sigma)
    #         conds.extend(chConds)
    #         inds.extend(chInds)
            
    #     return conds,inds
        
        
def arrXor(arr):
    val=False
    for a in arr:
        val=val^a
        
    return a

# def toParentIndex(nthChild,nthNode,nXparent,nXchild):
    
#     def ndx2arr(ndx,nX):
#         arr=[]
#         for ii in range(3):
#             arr.append(ndx%nX)
#             ndx=ndx//nX
#         return np.array(arr)
    
#     def arr2ndx(arr,nX):
#         mask=np.array([nX**n for n in range(3)])
#         return np.dot(arr,mask)
    
#     childArr=toBitArray(nthChild)
#     nodeArr=
    
#     childOffset=arr2ndx(childArr, nX)
    
    
#     maskCh=toBitArray(nthChild)
#     maskN=toBitArray(nthNode)
#     trips=np.array([3**n for n in range(3)])
#     return np.dot(trips,maskCh)+np.dot(trips,maskN)


def analyticVsrc(srcCoord,srcAmplitude,rVec,srcType='Current',sigma=1, srcRadius=1e-6):
    # r=np.linalg.norm(srcCoord-sampleCoords,axis=1)
    
    rVec[rVec<srcRadius]=srcRadius
    # if r<srcRadius:
    #     r=srcRadius
        
    if srcType=='Current':
        v0=srcAmplitude/(4*np.pi*sigma*srcRadius)
    else:
        v0=srcAmplitude
        
    return v0*srcRadius/rVec
    
# orig=np.random.rand(3)
# span=np.random.uniform(size=3)
# el=Element(orig,span,np.ones(3,dtype=np.float64))
# v=np.random.rand(8)
# interpC=getElementInterpolant(el,v)
# bbox=np.concatenate((orig,orig+span))
# ax=new3dPlot(bbox)
# coords=el.getCoordsRecursively()


# ivecs=[getCurrentVector(interpC,pt) for pt in coords]
# X,Y,Z=np.hsplit(coords,3)


# dx,dy,dz=np.hsplit(np.array(ivecs),3)

# showNodes(ax, coords, v)
# ax.quiver3D(X,Y,Z,dx,dy,dz)

# mid=orig+span/2

# iMid=getCurrentVector(interpC, mid)
# dx,dy,dz=np.hsplit(np.array(iMid),3)
# X,Y,Z=mid

# ax.quiver3D(X,Y,Z,dx,dy,dz)