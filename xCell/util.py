#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:40:44 2021
Utilities
@author: benoit
"""



import numpy as np
import numba as nb
from numba import int64, float64



nb.config.DISABLE_JIT=0
nb.config.DEBUG_TYPEINFER=0

# @nb.njit
def intify(pts,nX):
    mn=np.min(pts,axis=0)
    mx=np.max(pts,axis=0)
    
    span=mx-mn
    
    float0=pts-mn
    ints=nX*float0/span
    return ints.astype(np.int64)

# @nb.njit(parallel=True)
def uniformResample(coords,vals,nX):
    imgArray=np.empty((nX,nX),dtype=np.float64)
    ptI=intify(coords,nX)
    
    qVals,qIdx = getquads(ptI[:,0],ptI[:,1],
             ptI[:,0],ptI[:,1],vals)
    
    for ii in nb.prange(qVals.shape[0]):
        valset=qVals[ii]
        bnds=qIdx[ii]
        coef=getBilinCoefs(valset).squeeze()
        
        origin=bnds[[0,2]]
        span=bnds[[1,3]]-origin
        for xx in range(span[0]):
            for yy in range(span[1]):
                
                gX=xx+origin[0]
                gY=yy+origin[1]
                
                localCoord=np.array([xx,yy],dtype=np.float64)/span
                localCoef=toBilinearVars(localCoord)
                imgArray[gX,gY]=np.dot(coef,localCoef)
    
    return imgArray


def getquads(x,y,xInt,yInt,values):
    # x,y=np.hsplit(xy,2)
    # x=xy[:,0]
    # y=xy[:,1]
    _,kx,nx=np.unique(xInt,return_index=True,return_inverse=True)
    _,ky,ny=np.unique(yInt,return_index=True,return_inverse=True)
    
    quadVals, quadCoords = __getquadLoop(x,y,kx,ky,nx,ny,values)
            
    return quadVals, quadCoords

# @nb.njit(parallel=True)
def __getquadLoop(x,y,kx,ky,nx,ny,values):
    quadVals=[]
    quadCoords=[]
    
    sel=np.empty((4,x.shape[0]),dtype=np.bool8)
    
    for yy in nb.prange(len(ky)-1):
        for xx in nb.prange(len(kx)-1):
            
            sel=__getSel(nx, ny, xx, yy)
            ok=True
            for ii in np.arange(4):
                ok&=np.any(sel[ii])
            
            if ok:
                # x0,y0=xy[sel[0]][0]
                # x1,y1=xy[sel[3]][0]
                x0=x[sel[0]][0]
                y0=y[sel[0]][0]
                x1=x[sel[3]][0]
                y1=y[sel[3]][0]
                qcoords=np.array([x0,x1,y0,y1])
                where=np.array([np.nonzero(s)[0][0] for s in sel])
                qvals=values[where]
                # qvals=np.array([values[sel[n,:]] for n in np.arange(4)]).squeeze()

                quadVals.append(qvals)
                quadCoords.append(qcoords)
                
    return np.array(quadVals), np.array(quadCoords)

@nb.njit()
def __getSel(x,y,x0,y0,k=0):
    sel=np.empty((4,x.shape[0]),dtype=np.bool8)
    
    sel[0]=np.logical_and(x==x0,y==y0)
    sel[1]=np.logical_and(x>(x0+k),y==y0)
    sel[2]=np.logical_and(x==x0,y>(y0+k))
    
    if np.any(sel[1]) & np.any(sel[2]):
        x1=x[sel[1]][0]
        y1=y[sel[2]][0]
        
        sel[3]=np.logical_and(x==x1,y==y1)
        
        if not np.any(sel[3]):
            sel=__getSel(x, y, x0, y0,k=k+1)
        else: 
            sel[1]=np.logical_and(x==x1,y==y0)
            sel[2]=np.logical_and(x==x0,y==y1)
    
    
    return sel
   

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