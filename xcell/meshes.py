#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:43:26 2022

@author: benoit
"""
import numpy as np
import numba as nb
from . import elements
from . import util
from . import fem
from . import geometry as geo


class Mesh:
    def __init__(self, bbox, elementType='Admittance'):
        self.bbox = bbox
        self.extents = (bbox[3:]-bbox[:3])/2
        self.span = bbox[3:]-bbox[:3]
        self.center = bbox[:3]+self.span/2
        self.elements = []
        self.conductances = []
        self.elementType = elementType
        self.nodeCoords = np.empty((0, 3), dtype=np.float64)
        self.edges = []

        self.minl0 = 0

        self.indexMap = []
        self.inverseIdxMap = {}
        self.boundaryNodes = np.empty(0, dtype=np.int64)

    def __getstate__(self):

        state = self.__dict__.copy()

        elInfo = []
        for el in self.elements:
            d = {'origin': el.origin,
                 'span': el.span,
                 'l0': el.l0,
                 'sigma': el.sigma,
                 'vertices': el.vertices,
                 'faces': el.faces,
                 'index': el.index}
            elInfo.append(d)

        state['elements'] = elInfo
        state['inverseIdxMap'] = {}
        if 'tree' in state:
            dtree = {'origin': self.tree.origin,
                     'span': self.tree.span,
                     'sigma': self.tree.sigma,
                     'index': self.tree.index
                     }
            state['tree'] = dtree
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        elDicts = self.elements.copy()

        self.elements = []

        for ii, el in enumerate(elDicts):
            self.addElement(el['origin'], el['span'],
                            el['sigma'], el['index'])
            self.elements[-1].faces = el['faces']
            self.elements[-1].vertices = el['vertices']

        self.inverseIdxMap = util.getIndexDict(self.indexMap)

        if 'tree' in state:
            treeDict = self.tree.copy()
            tree = Octant(origin=treeDict['origin'],
                          span=treeDict['span'],
                          sigma=treeDict['sigma'],
                          index=treeDict['index'])
            tree._Octant__recreateTree(self.elements)
            self.tree = tree

    def getContainingElement(self, coords):
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
        nElem = len(self.elements)

        # TODO: workaround fudge factor
        tol = 1e-9*np.ones(3)

        for nn in nb.prange(nElem):
            elem = self.elements[nn]
            # if type(elem) is dict:
            #     delta=coords-elem['origin']
            #     ext=elem['extents']
            # else:
            delta = coords-elem.origin+tol
            ext = elem.span+2*tol

            difs = np.logical_and(delta >= 0, delta <= ext)
            if all(difs):
                return elem

        raise ValueError('Point (%s) not inside any element' %
                         ','.join(map(str, coords)))

    def finalize(self):
        pass

    def addElement(self, origin, span, sigma, index):
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
        #     newEl=elements.AdmittanceHex(origin,extents,sigma)
        # elif self.elementType=='FEM':
        #     newEl=elements.FEMHex(origin,extents,sigma)

        newEl = Octant(origin, span, sigma=sigma, index=index)
        # newEl.globalNodeIndices=nodeIndices
        self.elements.append(newEl)

    def getConductances(self, elements=None):
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
            elements = self.elements
        nElem = len(elements)
        # if self.elementType=='Admittance':
        #     nElemEdge=12
        # elif self.elementType=='FEM':
        #     nElemEdge=28
        # elif self.elementType=='Face':
        #     nElemEdge=6

        # nEdges=nElemEdge*nElem

        # conductances=np.empty(nEdges,dtype=np.float64)
        # edgeIndices=np.empty((nEdges,2),dtype=np.int64)

        clist = []
        elist = []
        transforms = []

        for nn in nb.prange(nElem):

            elem = elements[nn]

            # elConds=elem.getConductanceVals()
            # elEdges=elem.getConductanceIndices()
            if self.elementType == 'Admittance':
                elConds = fem.getAdmittanceConductances(elem.span, elem.sigma)
                elEdges = elem.vertices[fem.ADMITTANCE_EDGES]
            elif self.elementType == 'FEM':
                elConds = fem.getHexConductances(elem.span, elem.sigma)
                elEdges = elem.vertices[fem.getHexIndices()]
            elif self.elementType == 'Face':
                rawCond = fem.getFaceConductances(elem.span, elem.sigma)
                rawEdge = elem.faces[fem.FACE_EDGES]

                elConds = []
                elEdges = []
                for ii in nb.prange(6):
                    neighbors = elem.neighbors[ii]
                    nNei = len(neighbors)

                    if nNei <= 1:
                        elConds.append(rawCond[ii])
                        elEdges.append(rawEdge[ii])

                    else:
                        # faces are shared; split original edge
                        origNode = rawEdge[ii, 0]
                        xform = []

                        # neiNodeIdx=ii
                        neiNodeIdx = ii+(-1)**ii

                        # if elem.depth==11:
                        #     print()
                        for jj in nb.prange(nNei):
                            elConds.append(rawCond[ii]/nNei)

                            neighbor = neighbors[jj]
                            neiNode = neighbor.faces[neiNodeIdx]

                            edge = np.array([neiNode,
                                             rawEdge[ii, 1]])
                            elEdges.append(edge)
                            xform.append(neiNode)

                        xform.append(origNode)
                        transforms.append(xform)

            # conductances[nn*nElemEdge:(nn+1)*nElemEdge]=elConds
            # edgeIndices[nn*nElemEdge:(nn+1)*nElemEdge,:]=elEdges

            elist.extend(elEdges)
            clist.extend(elConds)

        # self.edges=edgeIndices

        edgeIndices = np.array(elist)
        conductances = np.array(clist)

        return edgeIndices, conductances, transforms

    def getL0Min(self):
        """
        Get the smallest edge length in mesh

        Returns
        -------
        l0Min : float
            smallest edge length.

        """
        l0Min = np.infty

        for el in self.elements:
            l0Min = min(l0Min, el.l0)
        return l0Min

    def getBoundaryNodes(self):
        mins, maxes = np.hsplit(self.bbox, 2)

        atmin = np.equal(mins, self.nodeCoords)
        atmax = np.equal(maxes, self.nodeCoords)
        isbnd = np.any(np.logical_or(atmin, atmax), axis=1)
        globalIndices = np.nonzero(isbnd)[0]

        return globalIndices

    def getIntersectingelements(self, axis, coordinate):
        elements = []

        for el in self.elements:
            gt = el.origin[axis] >= coordinate
            lt = (el.origin[axis]-el.span[axis]) < coordinate
            if gt and lt:
                elements.append(el)

        return elements


class Octree(Mesh):
    def __init__(self, boundingBox, maxDepth=10, elementType='Admittance'):
        self.center = np.mean(boundingBox.reshape(2, 3), axis=0)
        self.span = (boundingBox[3:]-boundingBox[:3])
        super().__init__(boundingBox, elementType)

        self.maxDepth = maxDepth
        self.bbox = boundingBox
        self.indexMap = np.empty(0, dtype=np.uint64)
        self.inverseIdxMap = {}

        self.changed = True

        # coord0=self.center-self.span/2

        self.tree = Octant(origin=boundingBox[:3],
                           span=self.span)

    def getContainingElement(self, coords):

        el = self.tree.getContainingElement(coords)
        return el

    def getIntersectingelements(self, axis, coordinate):
        elements = self.tree.getIntersectingElement(axis, coordinate)
        return elements

    def refineByMetric(self, minl0Function, refPts, maxl0Function=None, coefs=None):
        """
        Recursively splits elements until l0Function evaluated at the center
        of each element is greater than that element's l0'

        Parameters
        ----------
        l0Function : function
            Function returning a scalar for each input cartesian coordinate.

        Returns
        -------
        changed : bool
            Adaptation resulted in new topology

        """

        if maxl0Function is None:
            maxl0Function=minl0Function
        changed = self.tree.refineByMetric(minl0Function, refPts, self.maxDepth, coefs)
        pruned, _ = self.tree.coarsenByMetric(maxl0Function, refPts, self.maxDepth, coefs)

        # if pruned:
        #     print()
        #     changed|=pruned

        self.changed = changed | pruned

        return changed

    def finalize(self):
        """
        Convert terminal octants to mesh elements, mapping sparse to dense numbering.

        Returns
        -------
        None.

        """
        octs = self.tree.getTerminalOctants()

        # self.nodeCoords=self.getCoordsRecursively()

        # d=util.getIndexDict(self.indexMap)

        self.elements = octs

        if self.elementType == 'Face':
            self.getElementAdjacencies()

        # self.inverseIdxMap=d
        self.changed = False

        return

    def printStructure(self):
        """
        Debug tool to print structure of tree

        Returns
        -------
        None.

        """
        self.tree.printStructure()

    def octantByList(self, indexList, octant=None):
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
        head = indexList.pop(0)

        # #debug

        # if (head<0) or (head>8):
        #     print()

        # #enddebug

        if octant is None:
            octant = self.tree
        oc = octant.children[head]
        if len(oc.children) == 0:
            # terminal octant reached, match found
            return [oc]
        else:
            if len(indexList) == 0:
                # list exhausted at non-terminal octant
                return oc.children
            else:
                # continue recursion
                return self.octantByList(indexList, oc)

    def countelements(self):

        return self.tree.countelements()

    def getCoordsRecursively(self):
        """
        Determine coordinates of mesh nodes.

        Recurses through `Octant.getCoordsRecursively`

        Returns
        -------
        coords : 2d float array
            Cartesian coordinates of each mesh node.

        """

        # TODO: parallelize?
        # around 2x element creation time (not bad)
        # rest of function very fast

        asdual = self.elementType == 'Face'
        i = self.tree.getCoordsRecursively(self.maxDepth, asDual=asdual)
        # if asdual:
        #     self.getElementAdjacencies()

        indices = np.unique(np.array(i, dtype=np.uint64))
        self.indexMap = indices

        # c=util.indexToCoords(indices,self.span,self.maxDepth+int(asdual))
        # coords=c+self.bbox[:3]
        coords = util.indexToCoords(indices, self.bbox[:3], self.span)

        return coords

    def getBoundaryNodes(self, asdual=False):
        bnodes = []
        # if asdual:
        #     nX=2**self.maxDepth
        # else:
        #     nX=1+2**self.maxDepth

        # nX=1+2**(self.maxDepth+1)
        for ii in nb.prange(self.indexMap.shape[0]):
            nn = self.indexMap[ii]
            xyz = util.index2pos(nn, util.MAXPT)
            if np.any(xyz == 0) or np.any(xyz == util.MAXPT-1):
                bnodes.append(ii)

        return np.array(bnodes)
        # bnodes=util.octreeLoop_GetBoundaryNodesLoop(nX,self.indexMap)
        # return bnodes

    def getDualMesh(self):
        octs = self.tree.getTerminalOctants()
        self.elements = octs
        numel = len(self.elements)

        coords = np.empty((numel, 3), dtype=np.float64)
        edges = []
        conductances = []
        nodeIdx = []
        bnodes = []

        temp = []
        for ii in nb.prange(numel):
            el = self.elements[ii]

            elIndex = util.octantListToIndex(np.array(el.index),
                                             self.maxDepth)
            el.globalNodeIndices[0] = elIndex
            coords[ii] = el.center
            nodeIdx.append(elIndex)

            gEl = fem.getFaceConductances(el.span, el.sigma)

            neighborList = util.octantNeighborIndexLists(np.array(el.index))

            # if len(neighborList)!=6:
            #     neighborList=util.octantNeighborIndexLists(np.array(el.index))

        #     temp.append(neighborList)

        # for ii in nb.prange(numel):
        #     neighborList=temp[ii]

            if ii == (numel):
                print()

            isBnd = False

            # debug trap
            # if len(neighborList)!=6:
            #     print(ii)
            #     print(el.index)
            #     print(elIndex)

            for step in nb.prange(6):
                neighborI = neighborList[step]
                if len(neighborI) == 0:
                    isBnd = True
                else:
                    #     neighborI=el.getNeighborIndex(step)
                    # if neighborI is not None:
                    # neighbor exists
                    tstNeighbor = self.octantByList(neighborI)
                    nNeighbors = len(tstNeighbor)

                    if nNeighbors > 1:
                        ax = step//2
                        dr = step % 2
                        neighbors = [n for n in tstNeighbor if util.toBitArray(
                            n.index[-1])[ax] ^ dr]
                        nNeighbors = len(neighbors)
                    else:
                        neighbors = tstNeighbor

                    for neighbor in neighbors:
                        # neighborIndex=neighbor.globalNodeIndices[0]
                        neighborIndex = util.octantListToIndex(np.array(neighbor.index),
                                                               self.maxDepth)

                        gA = fem.getFaceConductances(neighbor.span,
                                                     neighbor.sigma)[step//2]
                        gB = gEl[step//2]/nNeighbors

                        gnet = 0.5*(gA*gB)/(gA+gB)

                        conductances.append(gnet)
                        edges.append([elIndex, neighborIndex])

            if isBnd:
                bnodes.append(elIndex)

        idxMap = np.array(nodeIdx)

        corrEdges = util.renumberIndices(np.array(edges), idxMap)
        self.boundaryNodes = util.renumberIndices(np.array(bnodes, ndmin=2),
                                                  idxMap).squeeze()

        return coords, idxMap, corrEdges, np.array(conductances)

    def getElementAdjacencies(self):
        numel = len(self.elements)

        adjacencies = []

        for ii in nb.prange(numel):
            el = self.elements[ii]

            maybeNeighbors = util.octantNeighborIndexLists(np.array(el.index))
            neighborSet = []
            for nn in nb.prange(6):
                nei = maybeNeighbors[nn]
                if len(nei) > 0:
                    tstEl = self.octantByList(nei)

                    if len(tstEl) > 1:
                        ax = nn//2
                        dr = nn % 2

                        # select elements on opposite side of
                        # sameFace=[((n>>ax)&1) for n in range(6)]
                        # neighbors=[tstEl[n] for n in sameFace]

                        neighbors = [n for n in tstEl if util.toBitArray(
                            n.index[-1])[ax] ^ dr]
                        # neighbors=[n for n in tstEl if util.toBitArray(n.index[-1])[ax]]

                    else:
                        neighbors = tstEl

                else:
                    neighbors = []

                neighborSet.append(neighbors)

            el.neighbors = neighborSet
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
    def __init__(self, origin, span, depth=0, sigma=np.ones(3), index=[], oXYZ=np.zeros(3, dtype=np.int32)):
        # super().__init__(origin, span, sigma)
        self.origin = origin
        self.span = span
        self.center = origin+span/2
        self.bbox=np.hstack((self.origin,self.origin+self.span))
        self.l0 = np.prod(span)**(1/3)

        self.children = []
        self.depth = depth
        self.index = index

        # rdepth=util.MAXDEPTH-depth

        # don't calculate indices here, since we might split later
        self.vertices = []
        self.faces = []

        self.sigma = sigma

        self.neighbors = 6*[[]]
        self.oXYZ = oXYZ

    def __recreateTree(self, elements):
        # childLists=8*[[]]
        # for el in elements:
        #     if el.depth==self.depth+1:
        #         #existing child
        #         self.children.append(el)
        #     else:
        #         elOct=el.index[self.depth]
        #         childLists[elOct].append(el)

        # if len(self.children)==8:
        #     chList=self.children
        #     chOrder=[ch.index[-1] for ch in chList]
        #     self.children=[ch for ch in chList for ii in chOrder if ch.index[-1]==ii]
        # else:
        #     for ii,ch in enumerate(self.children):
        #         ch.__recreateTree(childLists[ii])
        childLists = []
        D = self.depth
        for ind in range(8):
            childList = [el for el in elements if el.index[D] == ind]
            childLists.append(childList)

        self.split()
        for ii, ch in enumerate(self.children):
            clist = childLists[ii]
            if len(clist) == 1:
                el = clist[0]
                el.depth = len(el.index)
                self.children[ii] = el
            else:
                ch.__recreateTree(clist)

    def countelements(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum([ch.countelements() for ch in self.children])

    # @nb.njit(parallel=True)
    def split(self, division=np.array([0.5, 0.5, 0.5])):
        newSpan = self.span*division
        # scale=np.array(2**(util.MAXDEPTH-len(self.index)),dtype=np.int32)

        for ii in nb.prange(8):
            offset = newSpan*util.OCT_INDEX_BITS[ii]
            # offset=util.toBitArray(ii)*newSpan
            newOrigin = self.origin+offset
            newIndex = self.index.copy()
            newIndex.append(ii)
            # oxyz=np.array(self.oXYZ+scale*util.OCT_INDEX_BITS[ii],dtype=np.int32)
            self.children.append(Octant(origin=newOrigin,
                                        span=newSpan,
                                        depth=self.depth+1,
                                        index=newIndex))
            # oXYZ=oxyz))

        # return self.children
    # def getOwnCoords(self):
    #     return [self.origin+self.span*util.toBitArray(n) for n in range(8)]

    def getCoordsRecursively(self, maxdepth, asDual=False):
        # T=Logger('depth  %d'%self.depth, printStart=False)
        if len(self.children) == 0:
            # coords=self.getOwnCoords()
            # indices=self.globalNodeIndices

            if asDual:
                # indices=self.calcFaceTags()
                indices = self.faces.tolist()
            else:
                # indices=self.calcVertexTags()
                indices = self.vertices.tolist()
            # indices=self.calcFaceTags(maxdepth)
        else:
            indices = []

            for ch in self.children:
                i = ch.getCoordsRecursively(maxdepth, asDual)
                indices.extend(i)

        return indices

    # @nb.njit(parallel=True)
    def refineByMetric(self, l0Function, refPts, maxDepth, coefs):
        changed = False
        l0Target,whichPts=util.reduceFunctions(l0Function,refPts, self.bbox, coefs=coefs)
        nextPts=refPts[whichPts]

        if coefs is not None:
            nextCoefs=coefs[whichPts]
        else:
            nextCoefs=coefs

        if nextPts.shape[0]>0 and self.depth<maxDepth:
            if len(self.children)==0:
                changed=True
                self.split()
            for ii in nb.prange(8):
                changed|=self.children[ii].refineByMetric(l0Function,
                                                          nextPts,
                                                          maxDepth, nextCoefs)

        return changed

    def coarsenByMetric(self, metric, refPts, maxdepth, coefs):
        """
        If element and all children are smaller than target, delete children

        Parameters
        ----------
        metric : TYPE
            DESCRIPTION.
        refPts : TYPE
            DESCRIPTION.
        maxdepth : TYPE
            DESCRIPTION.
        coefs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        changed = False
        _,whichPts=util.reduceFunctions(metric, refPts, self.bbox, coefs=coefs, returnUnder=False)

        #let metric implicitly prune if maxdepth lowered
        #causes insufficient meshing otherwise
        # undersize=np.all(whichPts) #or self.depth<maxdepth
        undersize=np.all(whichPts) or self.depth<maxdepth


        if self.isTerminal():
            #end condition
            pass
        else:
            #recurse to end
            for ch in self.children:
                chChanged,chUnder=ch.coarsenByMetric(metric, refPts, maxdepth, coefs)
                changed |= chChanged
                undersize &= chUnder

            if undersize:
                changed=True
                for ch in self.children:
                    del ch
                self.children=[]
                self.calcIndices()

        return changed, undersize

    def prune(self):
        for ch in self.children:
            ch.prune()
            del ch

    def printStructure(self):
        """
        Print out octree structure.

        Returns
        -------
        None.

        """
        base = '> '*self.depth
        print(base+str(self.l0))

        for ch in self.children:
            ch.printStructure()

    def isTerminal(self):
        terminal = len(self.children) == 0
        if terminal and len(self.vertices) == 0:
            self.calcIndices()
        return terminal

    def containsPoint(self, coord):
        gt = np.greater_equal(coord, self.origin)
        lt = np.less_equal(coord-self.origin, self.span)
        return np.all(gt & lt)

    def intersectsPlane(self, axis, coord):
        cmin = self.origin[axis]
        cmax = cmin+self.span[axis]
        return (cmin <= coord) & (coord < cmax)

    def getTerminalOctants(self):
        """
        Get all childless octants.

        Returns
        -------
        list of Octants
            Childless octants (the actual elements of the mesh)

        """
        if self.isTerminal():
            return [self]
        else:

            descendants = []
            for ch in self.children:
                # if len(ch.children)==0:
                grandkids = ch.getTerminalOctants()

                if grandkids is not None:
                    descendants.extend(grandkids)
        return descendants

    def calcIndices(self):

        elList = np.array(self.index, dtype=np.int8)
        inds = util.indicesWithinOctant(elList, fem.HEX_POINT_INDICES)

        self.vertices = inds[:8]
        self.faces = inds[8:]

    def getContainingElement(self, coords):
        if len(self.children) == 0:
            # if self.containsPoint(coords):
            if geo.isInBBox(self.bbox, coords):
                return self
            else:
                return None
        else:
            for ch in self.children:
                tmp = ch.getContainingElement(coords)
                if tmp is not None:
                    return tmp
            return None

    def getIntersectingElement(self, axis, coord):
        # if (len(self.children)==0) and (self.intersectsPlane(axis, coord)):
        #     return [self]

        # else:
        #     descendants=[]
        #     for ch in self.children:
        #         intersects=ch.getIntersectingElement(axis,coord)
        #         if intersects is not None:
        #             descendants.extend(intersects)
        #     return descendants
        descendants = []
        if len(self.children) == 0:
            if self.intersectsPlane(axis, coord):
                return [self]
            else:
                return descendants
        else:

            for ch in self.children:
                intersects = ch.getIntersectingElement(axis, coord)
                if intersects is not None:
                    descendants.extend(intersects)

            return descendants

    def getNeighborIndex(self, direction):

        ownInd = np.array(self.index)

        return util.octantNeighborIndexList(ownInd, direction)

    def interpolateWithin(self, coordinates, values):
        coords = fem.toLocalCoords(coordinates, self.center, self.span)
        if values.shape[0] == 8:
            interp = fem.interpolateFromVerts(values, coords)
        else:
            interp = fem.interpolateFromFace(values, coords)

        return interp

    # TODO: fix hack of changing node indices
    def getUniversalVals(self, knownValues):

        # oldInds=self.globalNodeIndices

        allVals = np.empty(15, dtype=np.float64)
        allInds = np.empty(15, dtype=np.int64)

        indV = self.vertices
        indF = self.faces

        if knownValues.shape[0] == 8:
            vVert = knownValues
            vFace = fem.interpolateFromVerts(knownValues,
                                             fem.HEX_FACE_COORDS)
        else:
            vFace = knownValues
            vVert = fem.interpolateFromFace(knownValues,
                                            fem.HEX_VERTEX_COORDS)

        allInds = np.concatenate((indV, indF))
        allVals = np.concatenate((vVert, vFace))

        return allVals, allInds

    def getPlanarValues(self, globalValues, axis=2, coord=0.):
        zcoord = (coord-self.center[axis])/self.span[axis]

        inds = 3*[[-1., 1.]]
        inds[axis] = [zcoord]
        localCoords = np.array([[x, y, z] for z in inds[2]
                               for y in inds[1] for x in inds[0]])

        if globalValues.shape[0] == 8:
            planeVals = fem.interpolateFromVerts(globalValues, localCoords)
        else:
            planeVals = fem.interpolateFromFace(globalValues, localCoords)

        return planeVals


class MeshStats:
    def __init__(self):
        self.l0Min = 0.
        self.l0Max = 0.
        self.numEls = 0
        self.numPts = 0

        self.metricCoeffs = np.array([])
        self.metricCoeffNames = []
