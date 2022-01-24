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



# nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0

# insert subset in global: main[subIndices]=subValues
# get subset: subVals=main[boolMask]
# subInGlobalIndices=nonzero(boolMask)
# globalInSubIndices=(-ones()[boolMask])


# @nb.njit(parallel=True)
def eliminateRows(matrix):
    N=matrix.shape[0]
    # lo=scipy.sparse.tril(matrix,-1)
    lo=matrix.copy()
    lo.sum_duplicates()
    
    rowIdx,colIdx=lo.nonzero()
    _,count=np.unique(rowIdx,return_counts=True)
    
    
    tup = __eliminateRowsLoop(rowIdx,colIdx,count)
    v,r,c = tup

            
            
    # xmat=scipy.sparse.coo_matrix(tup,
    #                              shape=(N,N))
    xmat=scipy.sparse.coo_matrix((v, (r, c)),
                                 shape=(N,N))
    
    return xmat

# @nb.njit(nb.types.Tuple(nb.float64[:], nb.int64[:], nb.int64[:])(nb.int64[:], nb.int64[:], nb.int64[:]),
#          parallel=True)
# @nb.njit(parallel=True)
def __eliminateRowsLoop(rowIdx, colIdx, count):
    r=[]
    c=[]
    v=[]
    hanging=count<6
    for ii in nb.prange(count.shape[0]):
        if hanging[ii]:
            nconn=count[ii]-1
            cols=colIdx[rowIdx==ii]
            cols=cols[cols!=ii]
            
            
            vals=np.ones(nconn)/nconn
            
            if vals.shape!=cols.shape:
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
    return (v,r,c)
    # return tup
    

def getLinearMetric(minl0,maxl0,domainSize):
    rmin=np.sqrt(3)*minl0/2
    rmax=np.sqrt(3)*(domainSize-maxl0/2)
    k=(maxl0-minl0)/(rmax-rmin)
    b=minl0-k*rmin
    
    @nb.njit()
    def metric(coord):
        r=np.linalg.norm(coord)
        return k*r+b
    
    return metric


def deduplicateEdges(edges,conductances):
    N=max(edges.ravel())+1
    mat=scipy.sparse.coo_matrix((conductances,
                                 (edges[:,0], edges[:,1])),
                                shape=(N,N))
    
    dmat=mat.copy()+mat.transpose()
    
    dedup=scipy.sparse.tril(dmat,k=-1)
    
    newEdges=np.array(dedup.nonzero()).transpose()
    
    newConds=dedup.data
    
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
    whereSubset= np.empty_like(globalMask)
    nSubset=globalMask.nonzero()[0].shape[0]
    whereSubset[globalMask]=np.arange(nSubset)
    
    return whereSubset

# @nb.njit(parallel=True)
@nb.njit
def getIndexDict(globalMask):
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
    indexDict=dict()
    globalInd=np.nonzero(globalMask)[0]
    
    for ii in nb.prange(globalInd.shape[0]):
        ind=globalInd[ii]
        indexDict[ind]=ii
        
    return indexDict

@nb.njit(parallel=True)
def renumberIndices(edges,globalMask):
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
    hasValidNode=np.empty_like(edges,dtype=np.bool_)
    # hasValidNode=globalMask[edges]
    # for ii in range(2):
    for ii in nb.prange(edges.shape[1]):
        hasValidNode[:,ii]=globalMask[edges[:,ii]]
        
    if edges.shape[1]>1:
        isValid=np.logical_and(hasValidNode[:,0],
                           hasValidNode[:,1])
    else:
        isValid=hasValidNode[:,0]
    
    validEdges=edges[isValid]
    nValid=validEdges.shape[0]
    subNumberedEdges=np.empty((nValid,edges.shape[1]),
                              dtype=np.int64)


    ### dict-based approach, roughly 2x slower for 1e6 nodes,6e6 edges
    # subsInGlobal=condenseIndices(globalMask)
    
    # for nn in nb.prange(nValid):
    #     for jj in nb.prange(2):
    #         n=validEdges[nn,jj]
    #         subNumberedEdges[nn,jj]=subsInGlobal[n]
    
    

    ### array-based approach; high memory usage
    nodeDict=condenseIndices(globalMask)
    
    for nn in nb.prange(validEdges.shape[0]):
        for jj in nb.prange(validEdges.shape[1]):
            n=validEdges[nn,jj]
            subNumberedEdges[nn,jj]=nodeDict[n]
        
    
    ####global search method; appears to crash...
    # validIndex=np.nonzero(globalMask)[0]
    # for nn in nb.prange(nValid):
    #     for jj in nb.prange(2):
    #         n=validEdges[nn,jj]
    #         subNumberedEdges[nn,jj]=reindex(n,validIndex)
    
    
    return subNumberedEdges

# @nb.njit(nb.int64[:](nb.int64[:],nb.int64[:]))
@nb.njit(parallel=True)
def sparse2denseIndex(sparseVals,denseVals):
    """
    Get the indices where each sparseVal are within DenseVals.
    
    .. deprecated:: 0.0.1
        Use 
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
    idxList=np.empty_like(sparseVals, dtype=np.int64)
    for ii in nb.prange(sparseVals.shape[0]):
        sval=sparseVals[ii]
        idxList[ii]=reindex(sval,denseVals)
        
    return idxList

@nb.njit(parallel=True)
def octreeLoop_GetBoundaryNodesLoop(nX,indexMap):
    bnodes=[]
    for ii in nb.prange(indexMap.shape[0]):
        nn=indexMap[ii]
        xyz=index2pos(nn,nX)
        if np.any(xyz==0) or np.any(xyz==(nX-1)):
            bnodes.append(ii)
            
    return np.array(bnodes,dtype=np.int64)

@nb.njit()
def reindex(sparseVal,denseList):
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
    for n,val in enumerate(denseList):
        if val==sparseVal:
            return n
        
    # raise ValueError('%d not a member of denseList'%sparseVal)
    raise ValueError('not a member of denseList')

def coords2MaskedArrays(intCoords,edges,planeMask,vals):
    pcoords=np.ma.masked_array(intCoords, mask=~planeMask.repeat(2))
    edgeInPlane=np.all(planeMask[edges], axis=1)
    pEdges=np.ma.masked_array(edges, mask=~edgeInPlane.repeat(2))
    edgeLs=abs(np.diff(np.diff(pcoords[pEdges], axis=2), axis=1)).squeeze()
    
    span=pcoords.max()
    edgeSizes=getUnmasked(np.unique(edgeLs))
    
    arrays=[]
    for s in edgeSizes:
        nArray=span//s+1
        arr=np.nan*np.empty((nArray,nArray))
        
        #get 
        edgesThisSize=np.ma.array(pEdges, mask=(edgeLs!=s).repeat(2))
        nodesThisSize,nConn=np.unique(edgesThisSize.compressed(),return_counts=True)
        
        # nodesThisSize[nConn<2]=np.ma.masked
        # whichNodes=getUnmasked(nodesThisSize)
        whichNodes=nodesThisSize
        
        arrCoords=getUnmasked(pcoords[whichNodes])//s
        arrI,arrJ=np.hsplit(arrCoords, 2)
        
        arr[arrI.squeeze(),arrJ.squeeze()]=vals[whichNodes]
        arrays.append(np.ma.masked_invalid(arr))
        
    return arrays
        
        
def getUnmasked(maskArray):
    if len(maskArray.shape)>1:
        isValid=np.all(~maskArray.mask, axis=1)
    else:
        isValid=~maskArray.mask
    return maskArray.data[isValid]

def quadsToMaskedArrays(quadInds,quadVals):
    arrays=[]
    quadSize=quadInds[:,1]-quadInds[:,0]
    nmax=max(quadInds.ravel())
    
    sizes=np.unique(quadSize)
    
    for s in sizes:
        which=quadSize==s
        nGrid=nmax//s+1
        vArr=np.nan*np.empty((nGrid,nGrid))
        grid0s=quadInds[:,[0,2]][which]//s
        vgrids=quadVals[which]
        
        # if nGrid==9:
        #     print()
        
        for ii in range(vgrids.shape[0]):
            a,b=grid0s[ii]
            v=vgrids[ii]
            vArr[a,b]=v[0]
            vArr[a+1,b]=v[1]
            vArr[a,b+1]=v[2]
            vArr[a+1,b+1]=v[3]
            
        vmask=np.ma.masked_invalid(vArr)
        arrays.append(vmask)
    return arrays

#TODO: deprecate?
@nb.njit(parallel=True)
def edgeCurrentLoop(gList,edgeMat,dof2Global,vvec,gCoords,srcCoords):
    currents=np.empty_like(gList,dtype=np.float64)
    nEdges=gList.shape[0]
    edges=np.empty((nEdges,2,3),
                   dtype=np.float64)
    for ii in nb.prange(nEdges):
        g=gList[ii]
        dofEdge=edgeMat[ii]
        globalEdge=dof2Global[dofEdge]
        
        vs=vvec[dofEdge]
        dv=vs[1]-vs[0]

        if dv<0:
            dv=-dv
            globalEdge=np.array([globalEdge[1], globalEdge[0]])
        
        
        for pp in np.arange(2):
            p=globalEdge[pp]
            if p<0:
                c=srcCoords[-1-p]
            else:
                c=gCoords[p]
            
            edges[ii,pp]=c
            
        # i=g*dv
        
        currents[ii]=g*dv
        # edges[ii]=np.array(coords)
        
    return (currents, edges)

@nb.njit()
def edgeRoles(edges,nodeRoleTable):
    edgeRoles=np.empty_like(edges)
    for ii in nb.prange(edges.shape[0]):
        edgeRoles[ii]=nodeRoleTable[edges[ii]]
        
    return edgeRoles


#TODO: deprecate, unused?
@nb.njit()
def edgeNodesOfType(edges, nodeSelect):
    N=edges.shape[0]
    matches=np.empty(N,dtype=np.int64)

    for ii in nb.prange(N):
        e=edges[ii]
        matches[ii]=np.sum(e)
        
    return matches

#TODO: deprecate, unused?
def getquads(x,y,xInt,yInt,values):
    # x,y=np.hsplit(xy,2)
    # x=xy[:,0]
    # y=xy[:,1]
    _,kx,nx=np.unique(xInt,return_index=True,return_inverse=True)
    _,ky,ny=np.unique(yInt,return_index=True,return_inverse=True)
    
    quadVals, quadCoords = __getquadLoop(x,y,kx,ky,nx,ny,values)
            
    return quadVals, quadCoords

#TODO: deprecate, unused?
# @nb.njit(parallel=True)
def __getquadLoop(x,y,kx,ky,nx,ny,values):
    quadVals=[]
    quadCoords=[]
    
    sel=np.empty((4,x.shape[0]),dtype=np.bool8)
    
    for yy in nb.prange(len(ky)-1):
        for xx in nb.prange(len(kx)-1):
            
            indices=__getSel(nx, ny, xx, yy)

            if indices.shape[0]>0:
                # x0,y0=xy[sel[0]][0]
                # x1,y1=xy[sel[3]][0]
                x0=x[indices[0]]
                y0=y[indices[0]]
                x1=x[indices[3]]
                y1=y[indices[3]]
                qcoords=np.array([x0,x1,y0,y1])
                # where=np.array([np.nonzero(s)[0][0] for s in sel])
                qvals=values[indices]
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
def interpolateBilin(nodeVals,location):
    locCoef=toBilinearVars(location)
    
    interpCoefs=getBilinCoefs(nodeVals)
    
    interpVal=np.dot(interpCoefs,locCoef)
    return interpVal

@nb.njit()
def getBilinCoefs(vals):
    inv=np.array([[ 1.,  0.,  0.,  0.],
       [-1.,  1.,  0.,  0.],
       [-1.,  0.,  1.,  0.],
       [ 1., -1., -1.,  1.]])
    
    interpCoefs= inv @ vals
    return interpCoefs
    
@nb.njit
def minOver(target,vals):
    sel=vals>=target
    return min(vals[sel])

@nb.njit 
def maxUnder(target,vals):
    sel=vals<=target
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
    
    im=np.array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 1., -1., -1.,  1.,  0.,  0.,  0.,  0.],
       [ 1., -1.,  0.,  0., -1.,  1.,  0.,  0.],
       [ 1.,  0., -1.,  0., -1.,  0.,  1.,  0.],
       [-1.,  1.,  1., -1.,  1., -1., -1.,  1.]])
    # interpCoefs=np.matmul(im,nodeVals)
    interpCoefs=im @ nodeVals
    
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
        if any(np.isin(searchVals,el)):
            return True
    
    return False

@nb.njit()
def index2pos(ndx,dX):
    arr=[]
    for ii in range(3):
        arr.append(ndx%dX)
        ndx=ndx//dX
    return np.array(arr)

@nb.njit()
def pos2index(pos,dX):
    vals=np.array([dX**n for n in range(3)])
    tmp=np.dot(vals,pos)
    newNdx=int(np.rint(tmp))
    return newNdx