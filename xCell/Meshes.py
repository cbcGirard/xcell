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
import FEM

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
        
        self.indexMap=[]
        self.inverseIdxMap={}
        self.boundaryNodes=np.empty(0,dtype=np.int64)
        
        
    def __getstate__(self):
        
        state=self.__dict__.copy()
        
        elInfo=[]
        for el in self.elements:
            d={'origin':el.origin,
               'span':el.span,
               'l0':el.l0,
               'sigma':el.sigma,
               'nodeIndices':el.globalNodeIndices}
            elInfo.append(d)
            
        state['elements']=elInfo
        state['inverseIdxMap']={}
        if 'tree' in state:
            state['tree']=None
        return state
    
    def __setstate__(self,state):
        self.__dict__.update(state)
        elDicts=self.elements.copy()
        self.elements=[]
        
        for ii,el in enumerate(elDicts):
            self.addElement(el['origin'], el['span'],
                            el['sigma'], el['nodeIndices'])
            
        self.inverseIdxMap=util.getIndexDict(self.indexMap)
        
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
        # if self.elementType=='Admittance':
        #     newEl=Elements.AdmittanceHex(origin,extents,sigma)
        # elif self.elementType=='FEM':
        #     newEl=Elements.FEMHex(origin,extents,sigma)
            
        newEl=Octant(origin, extents,sigma=sigma,index=nodeIndices)
        # newEl.setGlobalIndices(nodeIndices)
        self.elements.append(newEl)
        
    def getConductances(self,elements=None):
        """
        Get the discrete conductances from every element.

        Returns
        -------
        edgeIndices : int64[:,:]
            List of node pairs spanned by each conductance.
        conductances : float
            Conductance in siemens.

        """
        
        if elements is None:
            elements=self.elements
        nElem=len(elements)
        # if self.elementType=='Admittance':
        #     nElemEdge=12
        # elif self.elementType=='FEM':
        #     nElemEdge=28
        # elif self.elementType=='Face':
        #     nElemEdge=6
            
        # nEdges=nElemEdge*nElem
        
        # conductances=np.empty(nEdges,dtype=np.float64)
        # edgeIndices=np.empty((nEdges,2),dtype=np.int64)
        
        clist=[]
        elist=[]
        
        for nn in nb.prange(nElem):
            
            elem=elements[nn]
            # elConds=elem.getConductanceVals()
            # elEdges=elem.getConductanceIndices()
            if self.elementType=='Admittance':
                elConds=FEM.getAdmittanceConductances(elem.span,elem.sigma)
                elEdges=elem.globalNodeIndices[FEM.getAdmittanceIndices()]
            elif self.elementType=='FEM':
                elConds=FEM.getHexConductances(elem.span,elem.sigma)
                elEdges=elem.globalNodeIndices[FEM.getHexIndices()]
            elif self.elementType=='Face':
                rawCond=FEM.getFaceConductances(elem.span,elem.sigma)
                rawEdge=elem.globalNodeIndices[FEM.getFaceIndices()]
                
                elConds=[]
                elEdges=[]
                for ii in nb.prange(6):
                    neighbors=elem.neighbors[ii]
                    nNei=len(neighbors)
                    
                    if nNei<=1:
                        elConds.append(rawCond[ii])
                        elEdges.append(rawEdge[ii])
                        
                    else:
                        neiNode=ii+(-1)**ii
                        for jj in nb.prange(nNei):
                            elConds.append(rawCond[ii]/nNei)
                            
                            neighbor=neighbors[jj]
                            
                            edge=np.array([neighbor.globalNodeIndices[neiNode],
                                           rawEdge[ii,1]])
                            elEdges.append(edge)
                        
            
            # conductances[nn*nElemEdge:(nn+1)*nElemEdge]=elConds
            # edgeIndices[nn*nElemEdge:(nn+1)*nElemEdge,:]=elEdges
            
            elist.extend(elEdges)
            clist.extend(elConds)
        
        # self.edges=edgeIndices
        
        edgeIndices=np.array(elist)
        conductances=np.array(clist)
        
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
        self.inverseIdxMap={}
        
        
        coord0=self.center-self.span/2
        
        self.tree=Octant(coord0,self.span)
        
    def getContainingElement(self, coords):
        
        el=self.tree.getContainingElement(coords)
        return el
        
    def getIntersectingElements(self,axis,coordinate):
        elements=self.tree.getIntersectingElement(axis, coordinate)
        return elements
        
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


        # T=util.Logger('coords')
        self.nodeCoords=self.getCoordsRecursively()
        # T.logCompletion()
        
        # T=util.Logger('loop')
        
        
        #TODO: Unify mapping logic; fails for dual mesh?
        d=util.getIndexDict(self.indexMap)

        # for ii in nb.prange(len(octs)):
        #     o=octs[ii]
        #     onodes=o.globalNodeIndices
            
        #     gnodes=np.array([d[n] for n in onodes])
        #     # self.addElement(o.origin,o.span,np.ones(3),gnodes)
        #     o.globalNodeIndices=gnodes
            # 
        # T.logCompletion()
        
        # T=util.Logger('copy')
        self.elements=octs
        
        
        if self.elementType=='Face':
            self.getElementAdjacencies()
        
        
        self.inverseIdxMap=d
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
        
        #debug
        
        if (head<0) or (head>8):
            print()
        
        #enddebug
        
        if octant is None:
            octant=self.tree
        oc=octant.children[head]
        if len(oc.children)==0:
            #terminal octant reached, match found
            return [oc]
        else:
            if len(indexList)==0:
                #list exhausted at non-terminal octant
                return oc.children
            else:
                #continue recursion
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
        
        #TODO: parallelize?
        #around 2x element creation time (not bad)
        #rest of function very fast
        
        asdual=self.elementType=='Face'
        i=self.tree.getCoordsRecursively(self.maxDepth,asDual=asdual)
        # if asdual:
        #     self.getElementAdjacencies()
            
        
        indices=np.unique(i)
        self.indexMap=indices
        

        c=util.indexToCoords(indices,self.span,self.maxDepth+int(asdual))
        coords=c+self.bbox[:3]

        return coords
    
    def getBoundaryNodes(self,asdual=False):
        bnodes=[]
        # if asdual:
        #     nX=2**self.maxDepth
        # else:
        #     nX=1+2**self.maxDepth
        
        nX=1+2**(self.maxDepth+1)
        for ii in nb.prange(self.indexMap.shape[0]):
            nn=self.indexMap[ii]
            xyz=util.index2pos(nn,nX)
            if np.any(xyz==0) or np.any(xyz==(nX-1)):
                bnodes.append(ii)
        
        return np.array(bnodes)
        # bnodes=util.octreeLoop_GetBoundaryNodesLoop(nX,self.indexMap)
        # return bnodes
    
    def getDualMesh(self):
        octs=self.tree.getTerminalOctants()
        self.elements=octs
        numel=len(self.elements)
        
        coords=np.empty((numel,3),dtype=np.float64)
        edges=[]
        conductances=[]
        nodeIdx=[]
        bnodes=[]
        
        temp=[]
        for ii in nb.prange(numel):
            el=self.elements[ii]
            # if ii==879:
            #     print()
            # elIndex=util.octantListToXYZ(np.array(el.index))
            elIndex=util.octantListToIndex(np.array(el.index),
                                           self.maxDepth)
            el.globalNodeIndices[0]=elIndex
            coords[ii]=el.center
            nodeIdx.append(elIndex)
            
            gEl=FEM.getFaceConductances(el.span, el.sigma)
            
            neighborList=util.octantNeighborIndexLists(np.array(el.index))
            
            # if len(neighborList)!=6:
            #     neighborList=util.octantNeighborIndexLists(np.array(el.index))
            
        #     temp.append(neighborList)
            
        # for ii in nb.prange(numel):
        #     neighborList=temp[ii]
            
            if ii==(numel):
                print()
            
            isBnd=False

            # debug trap
            # if len(neighborList)!=6:
            #     print(ii)
            #     print(el.index)
            #     print(elIndex)
            
            for step in nb.prange(6):
                neighborI=neighborList[step]
                if len(neighborI)==0:
                    isBnd=True
                else:
            #     neighborI=el.getNeighborIndex(step)
                # if neighborI is not None:
                    #neighbor exists
                    tstNeighbor=self.octantByList(neighborI)
                    nNeighbors=len(tstNeighbor)
                    
                    if nNeighbors>1:
                        ax=step//2
                        dr=step%2
                        neighbors=[n for n in tstNeighbor if util.toBitArray(n.index[-1])[ax]^dr]
                        nNeighbors=len(neighbors)
                    else:
                        neighbors=tstNeighbor
                    
                    for neighbor in neighbors:
                        # neighborIndex=neighbor.globalNodeIndices[0]
                        neighborIndex=util.octantListToIndex(np.array(neighbor.index), 
                                                             self.maxDepth)
                        
                        gA=FEM.getFaceConductances(neighbor.span,
                                                          neighbor.sigma)[step//2]
                        gB=gEl[step//2]/nNeighbors
                        
                        gnet=0.5*(gA*gB)/(gA+gB)
                        
                        conductances.append(gnet)
                        edges.append([elIndex,neighborIndex])

            if isBnd:
                bnodes.append(elIndex)
                
        idxMap=np.array(nodeIdx)
        
        corrEdges=util.renumberIndices(np.array(edges),idxMap)
        self.boundaryNodes=util.renumberIndices(np.array(bnodes,ndmin=2),
                                                idxMap).squeeze()
        
        return coords, idxMap, corrEdges, np.array(conductances)
            
                    
    def getDualMesh2(self):
        octs=self.tree.getTerminalOctants()
        self.elements=octs
        numel=len(self.elements)
        
        coords=np.empty((numel,3),dtype=np.float64)
        edges=[]
        conductances=[]
        nodeIdx=[]
        bnodes=[]
        
        temp=[]
        for ii in nb.prange(numel):
            el=self.elements[ii]
            elIndex=util.octantListToIndex(np.array(el.index),
                                           self.maxDepth)
            el.globalNodeIndices[0]=elIndex
            coords[ii]=el.center
            nodeIdx.append(elIndex)
            
            gEl=FEM.getFaceConductances(el.span, el.sigma)
            
            neighborList=util.octantNeighborIndexLists(np.array(el.index))
         
            isBnd=False

            
            for step in nb.prange(6):
                neighborI=neighborList[step]
                if len(neighborI)==0:
                    isBnd=True
                else:
                    #neighbor exists
                    tstNeighbor=self.octantByList(neighborI)
                    nNeighbors=len(tstNeighbor)
                    
                    if nNeighbors>1:
                        ax=step//2
                        dr=step%2
                        neighbors=[n for n in tstNeighbor if util.toBitArray(n.index[-1])[ax]^dr]
                        nNeighbors=len(neighbors)
                    else:
                        neighbors=tstNeighbor
                    
                    for neighbor in neighbors:
                        neighborIndex=util.octantListToIndex(np.array(neighbor.index), 
                                                             self.maxDepth)
                        
                        gA=FEM.getFaceConductances(neighbor.span,
                                                          neighbor.sigma)[step//2]
                        gB=gEl[step//2]/nNeighbors
                        
                        gnet=0.5*(gA*gB)/(gA+gB)
                        
                        conductances.append(gnet)
                        edges.append([elIndex,neighborIndex])

            if isBnd:
                bnodes.append(elIndex)
                
        idxMap=np.array(nodeIdx)
        
        corrEdges=util.renumberIndices(np.array(edges),idxMap)
        self.boundaryNodes=util.renumberIndices(np.array(bnodes,ndmin=2),
                                                idxMap).squeeze()
        
        return coords, idxMap, corrEdges, np.array(conductances)
                    
            
    def getElementAdjacencies(self):
        numel=len(self.elements)
        
        adjacencies=[]
        
        for ii in nb.prange(numel):
            el=self.elements[ii]
            
            maybeNeighbors=util.octantNeighborIndexLists(np.array(el.index))
            neighborSet=[]
            for nn in nb.prange(6):
                nei=maybeNeighbors[nn]
                if len(nei)>0:
                    tstEl=self.octantByList(nei)

                    if len(tstEl)>1:
                        ax=nn//2
                        dr=nn%2
                        neighbors=[n for n in tstEl if util.toBitArray(n.index[-1])[ax]^dr]
                    else:
                        neighbors=tstEl
                        
                else:
                    neighbors=[]
                        
                neighborSet.append(neighbors)
                
            el.neighbors=neighborSet
            adjacencies.append(neighborSet)
                    
        return adjacencies

# octant_t=nb.deferred_type()

# octantList_t=nb.deferred_type()

# octantspec= [
#     ('origin',nb.float64[:]),
#     ('span',nb.float64[:]),
#     ('center',nb.float64[:]),
#     ('l0',nb.float64),
#     ('children',nb.optional(octantList_t)),
#     ('depth',nb.int64),
#     ('globalNodeIndices',nb.int64[:]),
#     ('nX',nb.int64),
#     ('index',nb.int64[:]),
#     ('sigma',nb.float64[:])
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
        self.sigma=sigma
        
        self.neighbors=6*[[]]
        
    def calcVertexTags(self,maxdepth):
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
        return indices.tolist()
    
    def calcFaceTags(self,maxdepth):
        nX=2**(maxdepth+1)+1
        nInt=2**(maxdepth-self.depth)+1
        scale=2**(maxdepth-self.depth)
        
        # origin=np.zeros(3,dtype=np.int64)
        # for ii,idx in enumerate(self.index):
        #     ldepth=maxdepth-ii+1
        #     step=2**ldepth
        #     origin+=step*util.toBitArray(idx)
        origin=util.octantListToXYZ(np.array(self.index))*(scale<<1)
            
        ownstep=2**(maxdepth-self.depth+1)
        xyzScale=np.array([nX**n for n in range(3)])
        
        steps=np.array([[0,1,1],
                        [2,1,1],
                        [1,0,1],
                        [1,2,1],
                        [1,1,0],
                        [1,1,2],
                        [1,1,1]
                        ])*scale
        
        ptXYZ=[origin+step for step in steps]
        indices=np.array([np.dot(ndxlist,xyzScale) for ndxlist in ptXYZ])
            
        self.globalNodeIndices=indices
        
        return indices
            
        
    def countElements(self):
        if len(self.children)==0:
            return 1
        else:
            return sum([ch.countElements() for ch in self.children])
     
    # @nb.njit(parallel=True)
    def split(self,division=np.array([0.5,0.5,0.5])):
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

    
    def getCoordsRecursively(self,maxdepth,asDual=False):
        # T=Logger('depth  %d'%self.depth, printStart=False)
        if len(self.children)==0:
            # coords=self.getOwnCoords()
            # indices=self.globalNodeIndices
            
            if asDual:
                indices=self.calcFaceTags(maxdepth)
            else:
                indices=self.calcVertexTags(maxdepth)
            # indices=self.calcFaceTags(maxdepth)
        else:
            indices=[]
            
            for ch in self.children:
                i = ch.getCoordsRecursively(maxdepth,asDual)
                indices.extend(i)

        return indices
    

    
    # @nb.njit(parallel=True)
    def refineByMetric(self,l0Function,maxDepth):
        l0Target=l0Function(self.center)

        if (self.l0>l0Target) and (self.depth<maxDepth):

            self.split()
            for ii in nb.prange(8):
                self.children[ii].refineByMetric(l0Function,maxDepth)
                
    def printStructure(self):
        """
        Print out octree structure.

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
    
    def intersectsPlane(self,axis,coord):
        cmin=self.origin[axis]
        cmax=cmin+self.span[axis]
        return (cmin<=coord)&(coord<cmax)
    

        
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
        
    def getIntersectingElement(self,axis,coord):
        # if (len(self.children)==0) and (self.intersectsPlane(axis, coord)):
        #     return [self]

        # else:
        #     descendants=[]
        #     for ch in self.children:
        #         intersects=ch.getIntersectingElement(axis,coord)
        #         if intersects is not None:
        #             descendants.extend(intersects)
        #     return descendants
        descendants=[]
        if len(self.children)==0:
            if self.intersectsPlane(axis, coord):
                return [self]
            else:
                return descendants
        else:
            
            for ch in self.children:
                intersects=ch.getIntersectingElement(axis,coord)
                if intersects is not None:
                    descendants.extend(intersects)
                    
            return descendants
                
        
    def getNeighborIndex(self,direction):
                
        ownInd=np.array(self.index)

        return util.octantNeighborIndexList(ownInd, direction)
    
    
    def interpolateWithin(self,coordinates,values):
        coords=FEM.toLocalCoords(coordinates, self.center, self.span)
        if self.globalNodeIndices.shape[0]==8:
            interp=FEM.interpolateFromVerts(values, coords)
        else:
            interp=FEM.interpolateFromFace(values, coords)
        
        return interp
    
    def getPlanarValues(self,globalValues,axis=2,coord=0.):
        zcoord=(coord-self.center[axis])/self.span[axis]
        
        inds=3*[[-1.,1.]]
        inds[axis]=[zcoord]
        localCoords=np.array([[x,y,z] for z in inds[2] for y in inds[1] for x in inds[0]])
        
        if self.globalNodeIndices.shape[0]==8:
            planeVals=FEM.interpolateFromVerts(globalValues, localCoords)
        else:
            planeVals=FEM.interpolateFromFace(globalValues[:6], localCoords)
            
        return planeVals