#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:43:26 2022

@author: benoit
"""
import numpy as np
import numba as nb
import Elements
import util

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
            newEl=Elements.AdmittanceHex(origin,extents,sigma)
        elif self.elementType=='FEM':
            newEl=Elements.FEMHex(origin,extents,sigma)
            
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

        
    def finalize(self):
        """
        Convert terminal octants to mesh elements, mapping sparse to dense numbering.

        Returns
        -------
        None.

        """
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
        Select an octant by recursing through a list of indices.

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
        Determine coordinates of mesh nodes.
        
        Recurses through `Octant.getCoordsRecursively` 

        Returns
        -------
        coords : 2d float array
            Cartesian coordinates of each mesh node.

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
        # bnodes=util.octreeLoop_GetBoundaryNodesLoop(nX,self.indexMap)
        # return bnodes
    





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
    
    #TODO: deprecate unused def?
    # def getIndicesRecursively(self):
        
    #     if self.isTerminal():
    #         return np.arange(8,dtype=np.int64)
        
    #     else:
    #         indices=[]
    #         for ch in self.children:
    #             indices.append(ch.getIndicesRecursively())
    
    #         return indices
    
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
    
    #TODO: deprecate unused?
    # def distributeNodes(self,nodeCoords,nodeIndices):
    #     if self.isTerminal():
    #         return [], []
    #     else:
    #         for N in len(nodeIndices):
    #             if self.containsPoint(nodeCoords[N]):
    #                 self.innerNodes
                    
            
            
            
    #         return nodeCoords,nodeIndices
        
        
        
    def getTerminalOctants(self):
        """
        Get all childless octants.

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