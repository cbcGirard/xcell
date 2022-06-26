#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:40:44 2021
Utilities
@author: benoit
"""


import numpy as np
import numba as nb
import scipy
import time
import resource
import matplotlib.ticker as tickr

import psutil
from .fem import HEX_POINT_INDICES

# Atrocious typing hack to force use of uint64
MAXDEPTH = np.array(20, dtype=np.uint64)[()]
MAXPT = np.array(2**(MAXDEPTH+1)+1, dtype=np.uint64)[()]


# insert subset in global: main[subIndices]=subValues
# get subset: subVals=main[boolMask]
# subInGlobalIndices=nonzero(boolMask)
# globalInSubIndices=(-ones()[boolMask])

# @nb.njit()
def pointCurrentV(tstCoords, iSrc, sigma=1., srcLoc=np.zeros(3, dtype=np.float64)):
    dif = tstCoords-srcLoc
    k = 1./(4*np.pi*sigma)
    v = np.array([k/np.linalg.norm(d) for d in dif])
    return v


def makeGridPoints(nx, xmax, xmin=None, ymax=None, ymin=None, centers=False):
    if xmin is None:
        xmin = -xmax
    if ymax is None:
        ymax = xmax
    if ymin is None:
        ymin = -ymax

    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, nx)

    if centers:
        xx = 0.5*(xx[:-1]+xx[1:])
        yy = 0.5*(yy[:-1]+yy[1:])

    XX, YY = np.meshgrid(xx, yy)

    pts = np.vstack((XX.ravel(), YY.ravel())).transpose()
    return pts


@nb.experimental.jitclass()
class IndexMap:
    def __init__(self, sparseIndices):
        self.sparse = sparseIndices

    def __findMatch(self, value, lower, upper):
        if lower == upper:
            match = lower
        else:
            mid = (lower+upper)//2
            if self.sparse[mid] > value:
                match = self.__findMatch(value, lower, mid)
            else:
                match = self.__findMatch(value, mid, upper)

        return match

    def toDense(self, sparseIndex):
        match = self.__findMatch(sparseIndex, 0, self.sparse.shape[0])
        return match

# @nb.njit(parallel=True)


def eliminateRows(matrix):
    N = matrix.shape[0]
    # lo=scipy.sparse.tril(matrix,-1)
    lo = matrix.copy()
    lo.sum_duplicates()

    rowIdx, colIdx = lo.nonzero()
    _, count = np.unique(rowIdx, return_counts=True)

    tup = __eliminateRowsLoop(rowIdx, colIdx, count)
    v, r, c = tup

    # xmat=scipy.sparse.coo_matrix(tup,
    #                              shape=(N,N))
    xmat = scipy.sparse.coo_matrix((v, (r, c)),
                                   shape=(N, N))

    return xmat


def htile(array, ncopy):
    return np.vstack([array]*ncopy).transpose()

# @nb.njit(nb.types.Tuple(nb.float64[:], nb.int64[:], nb.int64[:])(nb.int64[:], nb.int64[:], nb.int64[:]),
#          parallel=True)
# @nb.njit(parallel=True)


def __eliminateRowsLoop(rowIdx, colIdx, count):
    r = []
    c = []
    v = []
    hanging = count < 6
    for ii in nb.prange(count.shape[0]):
        if hanging[ii]:
            nconn = count[ii]-1
            cols = colIdx[rowIdx == ii]
            cols = cols[cols != ii]

            vals = np.ones(nconn)/nconn

            if vals.shape != cols.shape:
                print()

            for jj in nb.prange(cols.shape[0]):
                c.append(cols[jj])
                r.append(ii)
                v.append(vals[jj])

        else:
            r.append(ii)
            c.append(ii)
            v.append(1.0)

    # tupIdx=(np.array(r),np.array(c))
    # tup=(np.array(v),tupIdx)
    # tup=(np.array(v), np.array(r), np.array(c))
    return (v, r, c)
    # return tup


def deduplicateEdges(edges, conductances):
    N = max(edges.ravel())+1
    mat = scipy.sparse.coo_matrix((conductances,
                                   (edges[:, 0], edges[:, 1])),
                                  shape=(N, N))

    dmat = mat.copy()+mat.transpose()

    dedup = scipy.sparse.tril(dmat, k=-1)

    newEdges = np.array(dedup.nonzero()).transpose()

    newConds = dedup.data

    return newEdges, newConds


@nb.njit
def condenseIndices(globalMask):
    """
    Get array for mapping local to global numbering.

    Parameters
    ----------
    globalMask : bool array
        Global elements included in subset.

    Returns
    -------
    whereSubset : int array
        DESCRIPTION.

    """
    # whereSubset=-np.ones_like(globalMask)
    whereSubset = np.empty_like(globalMask)
    nSubset = globalMask.nonzero()[0].shape[0]
    whereSubset[globalMask] = np.arange(nSubset)

    return whereSubset

# @nb.njit(parallel=True)


@nb.njit()
def getIndexDict(sparseIndices):
    """
    Get a dict of subsetIndex: globalIndex.

    .. deprecated:: 1.6.0
        Lower mem usage, but horribly slow.
        Use `condenseIndices` instead

    Parameters
    ----------
    globalMask : bool array
        Global elements included in subset.

    Returns
    -------
    indexDict : dict
        Dictionary of subset:global indices.

    """
    indexDict = {}
    for ii in range(sparseIndices.shape[0]):
        indexDict[sparseIndices[ii]] = ii

    return indexDict


def getPyDict(sparseIndices):
    dic = {}
    for ii, v in enumerate(sparseIndices):
        dic[v] = ii
    return dic


@nb.njit(parallel=True)
# @nb.vectorize([nb.int64(nb.int64, nb.int64)])
def renumberIndices(sparseIndices, denseList):
    """
    Renumber indices according to a subset.

    Parameters
    ----------
    edges : int array (1- or 2-d)
        Node indices by global numbering.
    globalMask : bool array
        Boolean mask of which global elements are in subset.

    Returns
    -------
    subNumberedEdges : int array
        Edges contained in subset, according to subset ordering.

    """
    renumbered = np.empty_like(sparseIndices, dtype=np.uint64)
    dic = getIndexDict(denseList)

    if sparseIndices.ndim == 1:
        for ii in nb.prange(sparseIndices.shape[0]):
            renumbered[ii] = dic[sparseIndices[ii]]
    else:
        for ii in nb.prange(sparseIndices.shape[0]):
            for jj in nb.prange(sparseIndices.shape[1]):
                renumbered[ii, jj] = dic[sparseIndices[ii, jj]]

    return renumbered

    # dic=getIndexDict(denseList)
    # renumbered=dic[sparseIndices]

    # return renumbered


@nb.njit(parallel=True)
def octreeLoop_GetBoundaryNodesLoop(nX, indexMap):
    bnodes = []
    for ii in nb.prange(indexMap.shape[0]):
        nn = indexMap[ii]
        xyz = index2pos(nn, nX)
        if np.any(xyz == 0) or np.any(xyz == (nX-1)):
            bnodes.append(ii)

    return np.array(bnodes, dtype=np.int64)


@nb.njit()
def reindex(sparseVal, denseList):
    """
    Get position of sparseVal in denseList, returning as soon as found.

    Parameters
    ----------
    sparseVal : int64
        Value to find index of match.
    denseList : int64[:]
        List of nonconsecutive indices.

    Raises
    ------
    ValueError
        Error if sparseVal not found.

    Returns
    -------
    int64
        index where sparseVal occurs in denseList.

    """
    # startguess=sparseVal//denseList[-1]
    # for n,val in np.ndenumerate(denseList):
    for n, val in enumerate(denseList):
        if val == sparseVal:
            return n

    # raise ValueError('%d not a member of denseList'%sparseVal)
    raise ValueError('not a member of denseList')


def coords2MaskedArrays(intCoords, edges, planeMask, vals):
    pcoords = np.ma.masked_array(intCoords, mask=~planeMask.repeat(2))
    edgeInPlane = np.all(planeMask[edges], axis=1)
    pEdges = np.ma.masked_array(edges, mask=~edgeInPlane.repeat(2))
    edgeLs = abs(np.diff(np.diff(pcoords[pEdges], axis=2), axis=1)).squeeze()

    span = pcoords.max()
    edgeSizes = getUnmasked(np.unique(edgeLs))

    arrays = []
    for s in edgeSizes:
        nArray = span//s+1
        arr = np.nan*np.empty((nArray, nArray))

        # get
        edgesThisSize = np.ma.array(pEdges, mask=(edgeLs != s).repeat(2))
        nodesThisSize, nConn = np.unique(
            edgesThisSize.compressed(), return_counts=True)

        # nodesThisSize[nConn<2]=np.ma.masked
        # whichNodes=getUnmasked(nodesThisSize)
        whichNodes = nodesThisSize

        arrCoords = getUnmasked(pcoords[whichNodes])//s
        arrI, arrJ = np.hsplit(arrCoords, 2)

        arr[arrI.squeeze(), arrJ.squeeze()] = vals[whichNodes]
        arrays.append(np.ma.masked_invalid(arr))

    return arrays


def getUnmasked(maskArray):
    if len(maskArray.shape) > 1:
        isValid = np.all(~maskArray.mask, axis=1)
    else:
        isValid = ~maskArray.mask
    return maskArray.data[isValid]


def quadsToMaskedArrays(quadInds, quadVals):
    arrays = []
    quadSize = quadInds[:, 1]-quadInds[:, 0]
    nmax = max(quadInds.ravel())

    sizes = np.unique(quadSize)

    for s in sizes:
        which = quadSize == s
        nGrid = nmax//s+1
        vArr = np.nan*np.empty((nGrid, nGrid))
        grid0s = quadInds[:, [0, 2]][which]//s
        vgrids = quadVals[which]

        # if nGrid==9:
        #     print()

        for ii in range(vgrids.shape[0]):
            a, b = grid0s[ii]
            v = vgrids[ii]
            vArr[a, b] = v[0]
            vArr[a+1, b] = v[1]
            vArr[a, b+1] = v[2]
            vArr[a+1, b+1] = v[3]

        vmask = np.ma.masked_invalid(vArr)
        arrays.append(vmask)
    return arrays

# TODO: deprecate?


@nb.njit(parallel=True)
def edgeCurrentLoop(gList, edgeMat, dof2Global, vvec, gCoords, srcCoords):
    currents = np.empty_like(gList, dtype=np.float64)
    nEdges = gList.shape[0]
    edges = np.empty((nEdges, 2, 3),
                     dtype=np.float64)
    for ii in nb.prange(nEdges):
        g = gList[ii]
        dofEdge = edgeMat[ii]
        globalEdge = dof2Global[dofEdge]

        vs = vvec[dofEdge]
        dv = vs[1]-vs[0]

        if dv < 0:
            dv = -dv
            globalEdge = np.array([globalEdge[1], globalEdge[0]])

        for pp in np.arange(2):
            p = globalEdge[pp]
            if p < 0:
                c = srcCoords[-1-p]
            else:
                c = gCoords[p]

            edges[ii, pp] = c

        # i=g*dv

        currents[ii] = g*dv
        # edges[ii]=np.array(coords)

    return (currents, edges)


@nb.njit()
def edgeRoles(edges, nodeRoleTable):
    edgeRoles = np.empty_like(edges)
    for ii in nb.prange(edges.shape[0]):
        edgeRoles[ii] = nodeRoleTable[edges[ii]]

    return edgeRoles


# TODO: deprecate, unused?
@nb.njit()
def edgeNodesOfType(edges, nodeSelect):
    N = edges.shape[0]
    matches = np.empty(N, dtype=np.int64)

    for ii in nb.prange(N):
        e = edges[ii]
        matches[ii] = np.sum(e)

    return matches

# TODO: deprecate, unused?


def getquads(x, y, xInt, yInt, values):
    # x,y=np.hsplit(xy,2)
    # x=xy[:,0]
    # y=xy[:,1]
    _, kx, nx = np.unique(xInt, return_index=True, return_inverse=True)
    _, ky, ny = np.unique(yInt, return_index=True, return_inverse=True)

    quadVals, quadCoords = __getquadLoop(x, y, kx, ky, nx, ny, values)

    return quadVals, quadCoords

# TODO: deprecate, unused?
# @nb.njit(parallel=True)


def __getquadLoop(x, y, kx, ky, nx, ny, values):
    quadVals = []
    quadCoords = []

    sel = np.empty((4, x.shape[0]), dtype=np.bool8)

    for yy in nb.prange(len(ky)-1):
        for xx in nb.prange(len(kx)-1):

            indices = __getSel(nx, ny, xx, yy)

            if indices.shape[0] > 0:
                # x0,y0=xy[sel[0]][0]
                # x1,y1=xy[sel[3]][0]
                x0 = x[indices[0]]
                y0 = y[indices[0]]
                x1 = x[indices[3]]
                y1 = y[indices[3]]
                qcoords = np.array([x0, x1, y0, y1])
                # where=np.array([np.nonzero(s)[0][0] for s in sel])
                qvals = values[indices]
                # qvals=np.array([values[sel[n,:]] for n in np.arange(4)]).squeeze()

                quadVals.append(qvals)
                quadCoords.append(qcoords)

    return np.array(quadVals), np.array(quadCoords)


@nb.njit
def toBilinearVars(coord):
    return np.array([1.,
                     coord[0],
                     coord[1],
                     coord[0]*coord[1]])


@nb.njit
def interpolateBilin(nodeVals, location):
    locCoef = toBilinearVars(location)

    interpCoefs = getBilinCoefs(nodeVals)

    interpVal = np.dot(interpCoefs, locCoef)
    return interpVal


@nb.njit()
def getBilinCoefs(vals):
    inv = np.array([[1.,  0.,  0.,  0.],
                    [-1.,  1.,  0.,  0.],
                    [-1.,  0.,  1.,  0.],
                    [1., -1., -1.,  1.]])

    interpCoefs = inv @ vals
    return interpCoefs


@nb.njit
def minOver(target, vals):
    sel = vals >= target
    return min(vals[sel])


@nb.njit
def maxUnder(target, vals):
    sel = vals <= target
    return max(vals[sel])


# @nb.njit
# def getElementInterpolant(element,nodeVals):
@nb.njit()
def getElementInterpolant(nodeVals):
    # coords=element.getOwnCoords()
    # xx=np.arange(2,dtype=np.float64)
    # # coords=np.array([[x,y,z] for z in xx for y in xx for x in xx])
    # coords=np.array([[0.,0.,0.],
    #                  [1.,0.,0.],
    #                  [0.,1.,0.],
    #                  [1.,1.,0.],
    #                  [0.,0.,1.],
    #                  [1.,0.,1.],
    #                  [0.,1.,1.],
    #                  [1.,1.,1.]])
    # coefs=np.empty((8,8),dtype=np.float64)
    # for ii in range(8):
    #     coefs[ii]=coord2InterpVals(coords[ii])

    # interpCoefs=np.linalg.solve(coefs, nodeVals)

    im = np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                   [1., -1., -1.,  1.,  0.,  0.,  0.,  0.],
                   [1., -1.,  0.,  0., -1.,  1.,  0.,  0.],
                   [1.,  0., -1.,  0., -1.,  0.,  1.,  0.],
                   [-1.,  1.,  1., -1.,  1., -1., -1.,  1.]])
    # interpCoefs=np.matmul(im,nodeVals)
    interpCoefs = im @ nodeVals

    return interpCoefs


@nb.njit
def evalulateInterpolant(interp, location):

    # coeffs of a, bx, cy, dz, exy, fxz, gyz, hxyz
    varList = coord2InterpVals(location)

    # interpVal=np.matmul(interp,varList)
    interpVal = np.dot(interp, varList)

    return interpVal


@nb.njit
def coord2InterpVals(coord):
    x, y, z = coord
    return np.array([1,
                     x,
                     y,
                     z,
                     x*y,
                     x*z,
                     y*z,
                     x*y*z]).transpose()


@nb.njit
def getCurrentVector(interpolant, location):
    # coeffs are
    # 0  1   2   3    4    5    6    7
    #a, bx, cy, dz, exy, fxz, gyz, hxyz
    # gradient is [
    #   [b + ey + fz + hyz],
    #   [c + ex + gz + hxz],
    #   [d + fx + gy + hxy]

    varList = coord2InterpVals(location)

    varSets = np.array([[0, 2, 3, 6],
                        [0, 1, 3, 5],
                        [0, 1, 2, 4]])
    coefSets = np.array([[1, 4, 5, 7],
                         [2, 4, 6, 7],
                         [3, 5, 6, 7]])

    varVals = np.array([varList[n] for n in varSets])
    coefVals = np.array([interpolant[n] for n in coefSets])

    vecVals = np.array([-np.dot(v, c) for v, c in zip(varVals, coefVals)])

    return vecVals


@nb.njit()  # ,parallel=True)
# @nb.njit(['int64[:](int64, int64)', 'int64[:](int64, Omitted(int64))'])
def toBitArray(val, nBits=3):
    return np.array([(val >> n) & 1 for n in range(nBits)])


OCT_INDEX_BITS = np.array([toBitArray(ii) for ii in range(8)])


@nb.njit()
def fromBitArray(arr):
    val = 0
    nbit = arr.shape[0]
    for ii in nb.prange(nbit):
        val += arr[ii]*2**(nbit-ii-1)
    return val


@nb.njit
def anyMatch(searchArray, searchVals):
    """
    Rapid search if any matches occur (returns immediately at first match).

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
        if any(np.isin(searchVals, el)):
            return True

    return False


@nb.njit()
def index2pos(ndx, dX):
    """
    Convert scalar index to [x,y,z] indices.

    Parameters
    ----------
    ndx : uint64
        Index to convert.
    dX : uint64
        Number of points per axis.

    Returns
    -------
    int64[:]
        DESCRIPTION.

    """
    arr = np.empty(3, dtype=np.uint64)
    for ii in range(3):
        a, b = divmod(ndx, dX)
        arr[ii] = b
        ndx = a

        # factor=dX**(2-ii)

        # val=
        # i,r=divmod(ndx,factor)
        # arr[2-ii]=i
        # ndx-=r*factor

    return arr


@nb.njit()
def pos2index(pos, dX):
    """
    Convert [x,y,z] indices to a scalar index

    Parameters
    ----------
    pos : int64[:]
        [x,y,z] indices.
    dX : int64
        Number of points per axis.

    Returns
    -------
    newNdx : int64
        Scalar index equivalent to [x,y,z] triple.

    """
    vals = np.array([dX**n for n in range(3)], dtype=np.uint64)
    # tmp=np.dot(vals,pos)
    # newNdx=int(np.rint(tmp))

    newNdx = intdot(pos, vals)

    return newNdx


@nb.njit()  # ,parallel=True)
def intdot(a, b):
    dot = np.empty(a.shape[0], dtype=np.uint64)

    for ii in nb.prange(a.shape[0]):
        dot[ii] = np.sum(a[ii]*b)

    return dot


@nb.njit()
def indicesWithinOctant(elList, relativePos):
    # origin=octantListToXYZ(elList)

    # npt=relativePos.shape[0]
    # depth=20-elList.shape[0]
    # scale=2**depth

    # indices=np.empty(npt,dtype=np.int64)

    # for ii in nb.prange(npt):
    #     pos=origin+scale*relativePos[ii]
    #     indices[ii]=pos2index(pos,nX)

    # absPos=np.empty_like(relativePos,dtype=np.uint64)
    # for ii in nb.prange(npt):
    #     for jj in nb.prange(3):
    #         absPos[ii,jj]=origin[jj]+scale*relativePos[ii,jj]

    absPos = xyzWithinOctant(elList, relativePos)

    # np.array([origin+scale*pos for pos in relativePos])
    indices = pos2index(absPos, MAXPT)

    return indices


# @nb.njit(parallel=True)
# def bulkCalcOctantIndices(elLists,elDepths):
#     numel=elLists.shape[0]
#     inds=np.empty((numel,15),dtype=np.uint64)
#     for nel in nb.prange(numel):
#         elList=elLists[nel,:elDepths[nel]]
#         inds[nel,:]=indicesWithinOctant(elList,
#                                        HEX_POINT_INDICES)

#     return inds

@nb.njit(parallel=True)
def xyzWithinOctant(elList, relativePos):
    origin = octantListToXYZ(elList)

    npt = relativePos.shape[0]
    depth = MAXDEPTH-elList.shape[0]
    scale = 2**depth

    absPos = np.empty_like(relativePos, dtype=np.uint64)
    for ii in nb.prange(npt):
        for jj in nb.prange(3):
            absPos[ii, jj] = origin[jj]+scale*relativePos[ii, jj]

    return absPos


@nb.njit(parallel=True,)
def indexToCoords(indices, origin, span):
    """


    Parameters
    ----------
    indices : TYPE
        DESCRIPTION.
    origin : TYPE
        DESCRIPTION.
    span : TYPE
        DESCRIPTION.
    maxDepth : TYPE
        DESCRIPTION.

    Returns
    -------
    coords : TYPE
        DESCRIPTION.

    """

    nPt = indices.shape[0]
    coords = np.empty((nPt, 3), dtype=np.float64)
    for ii in nb.prange(nPt):
        ijk = index2pos(indices[ii], MAXPT)
        coords[ii] = span*ijk/(MAXPT-1)+origin

    return coords

# @nb.njit(parallel=True)


@nb.njit()  # ,parallel=True)
def octantListToXYZ(octList):
    """


    Parameters
    ----------
    octList : TYPE
        DESCRIPTION.

    Returns
    -------
    XYZ : TYPE
        DESCRIPTION.

    """
    depth = octList.shape[0]
    xyz = np.zeros(3, dtype=np.uint64)
    for ii in nb.prange(depth):
        scale = 2**(MAXDEPTH-ii)
        ind = octList[ii]
        for ax in nb.prange(3):
            if (ind >> ax) & 1:
                xyz[ax] += scale

    return xyz


@nb.njit()
def uIndexToXYZ(index):

    return index2pos(index, MAXPT)

# @nb.njit(parallel=True)
# @nb.njit()
# def octantListToIndex(octList):
#     '''


#     Parameters
#     ----------
#     octList : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     index : TYPE
#         DESCRIPTION.

#     '''
#     index=0
#     listdepth=octList.shape[0]
#     k=21-listdepth

#     for ax in nb.prange(3):
#         shift=(ax+1)*21-1
#         for jj in nb.prange(listdepth):
#             bit=(octList[jj]>>ax)&1
#             index|=bit<<(shift-jj)

#             # if (octList[jj]//(2**ax))%2:
#             #     index+=2**(shift-jj)


#     return index

# @nb.njit(parallel=True)
# @nb.njit(parallel=True)
def octantNeighborIndexLists(ownIndexList):
    '''


    Parameters
    ----------
    ownIndexList : TYPE
        DESCRIPTION.

    Returns
    -------
    neighborLists : TYPE
        DESCRIPTION.

    '''
    ownDepth = ownIndexList.shape[0]
    nX = 2**ownDepth
    # nXYZ=octantListToXYZ(ownIndexList)
    nXYZ = octListReverseXYZ(ownIndexList)
    neighborLists = []
    # keep Numba type inference happy
    nullList = [np.int64(x) for x in range(0)]

    # bnds=[0,nX]

    for nn in nb.prange(6):
        axis = nn//2
        dx = nn % 2
        # if nXYZ[axis]==bnds[dx]:
        # if (nXYZ[axis]==0) or (nXYZ[axis]==(nX-1)):
        #     neighborLists.append(nullList)
        # else:
        #     dr=2*(dx)-1
        #     xyz=nXYZ.copy()
        #     xyz[axis]+=dr
        dr = 2*dx-1
        xyz = nXYZ.copy()
        xyz[axis] += dr

        neighborLists.append(__xyzToOList(xyz, ownDepth))

        # if np.any(xyz<0) or np.any(xyz>nX):
        #     neighborLists.append(nullList)
        # else:

        #     idxList=[]
        #     for ii in nb.prange(ownDepth):
        #         ind=0
        #         #HERE
        #         shift=ownDepth-ii-1
        #         mask=1<<shift
        #         for jj in nb.prange(3):
        #             bit=(xyz[jj]&mask)>>shift
        #             ind|=(bit<<jj)

        #         # mask=2**shift
        #         # for jj in nb.prange(3):
        #         #     bit=(xyz[jj]&mask)//shift
        #         #     ind+=bit*2**jj

        #         idxList.append(ind)
        #     neighborLists.append(idxList)

        # if len(idxList)!=6:
        #     print(nn)
        #     # print(neighborLists)
        #     # print("dammit")

    return neighborLists


@nb.njit()  # ,parallel=True)
def octListReverseXYZ(octantList):
    depth = octantList.shape[0]

    xyz = np.zeros(3, dtype=np.int64)
    for nn in nb.prange(depth):
        for jj in nb.prange(3):
            bit = (octantList[nn] >> jj) & 1
            # xyz[jj]+=bit<<nn
            xyz[jj] += bit << (depth-nn-1)

    return xyz


@nb.njit()  # ,parallel=True)
def __xyzToOList(xyz, depth):
    # keep Numba type inference happy
    nullList = [np.int64(x) for x in nb.prange(0)]
    xMax = 2**depth

    if np.any(xyz >= xMax) or np.any(xyz < 0):
        return nullList
    else:
        olist = []
        XYZ = xyz.astype(np.uint32)
        for nn in nb.prange(depth):
            k = 0
            for jj in nb.prange(3):
                bit = (XYZ[jj] >> nn) & 1
                # k+=bit<<(nn-jj)
                k += bit << jj
            olist.append(k)

        olist.reverse()
        return olist


class Logger():
    def __init__(self, stepName, printout=False):
        self.name = stepName
        self.printout = printout
        if printout:
            print(stepName+" starting")
        self.startWall = time.monotonic()
        self.start = time.process_time()
        self.durationCPU = 0
        self.durationWall = 0
        self.memory = 0

    def logCompletion(self):
        tWall = time.monotonic()
        tend = time.process_time()
        durationCPU = tend-self.start
        durationWall = tWall-self.startWall

        if self.printout:
            engFormat = tickr.EngFormatter()
            print(self.name+": "+engFormat(durationCPU) +
                  "s [CPU], "+engFormat(durationWall)+'s [wall]')
        self.durationCPU = durationCPU
        self.durationWall = durationWall
        # self.memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.memory = psutil.Process().memory_info().rss


def unravelArraySet(maskedArrays):
    vals = []
    for arr in maskedArrays:
        goodvals = arr.data[~arr.mask]
        vals.extend(goodvals.ravel())

    return np.array(vals)
