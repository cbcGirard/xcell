#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:37:11 2022

@author: benoit
"""

import numpy as np
import numba as nb
import util


@nb.njit()
def getHexConductances(span,sigma):
    if sigma.shape[0]==1:
        sigma=sigma*np.ones(3)
    else:
        sigma=sigma
        
    # k=self.span/(36*np.roll(self.span,1)*np.roll(self.span,2))
    k=np.roll(span,1)*np.roll(span,2)/(36*span)
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

@nb.njit()
def getHexIndices():
    edges=np.empty((28,2),dtype=np.int64)
    nn=0
    for ii in range(8):
        for jj in range(ii+1,8):
            edges[nn,:]=np.array([ii,jj])
            nn+=1
            
    return edges

@nb.njit()
def getFaceConductances(span,sigma):
    if sigma.shape[0]==1:
        sigma=sigma*np.ones(3)
    else:
        sigma=sigma
        
    k=np.roll(span,1)*np.roll(span,2)/span
    g=sigma*k*2
    return g.repeat(2)
        
        
@nb.njit()
def getFaceIndices():
    subsets=np.array([
        [2,3],
        [3,4],
        [1,3],
        [3,5],
        [0,3],
        [3,6]
        ],dtype=np.int64)
    return subsets


@nb.njit()
def getTetConductance(pts):
    g=np.empty(6,dtype=np.float64)
    pairs=getTetIndices()
    for ii in range(6):
        
        edge=pairs[ii]
        # others=~np.isin(np.arange(4),edge)
        others=pairs[5-ii]
        otherPts=pts[others]
        # othr=others(edge,pts)

        
    #     M=others(edge[1],pts)-pts[edge[1]]
        M=np.vstack((otherPts[1],pts[edge[0]], pts[edge[1]]))-otherPts[0]
        # [A,B,C]=vectorize(np.transpose(M))
        A=M[0,:]
        B=M[1,:]
        C=M[2,:]
    #     [A,B,C]=vectorize(M)
    
        num=np.dot(A,B)*np.dot(A,C)-np.dot(A,A)*np.dot(B,C)
        den=np.abs(6*np.linalg.det(M))
        g[ii]=-num/den
        
    return g

@nb.njit()
def getTetIndices():
    edges=np.array([[0,1],
                    [0,2],
                    [0,3],
                    [1,2],
                    [1,3],
                    [2,3]],dtype=np.int64)
    return edges

@nb.njit()
def getAdmittanceConductances(span,sigma):
    if sigma.shape[0]==1:
        sigma=sigma*np.ones(3)
    else:
        sigma=sigma
        
    k=np.roll(span,1)*np.roll(span,2)/span
    K=sigma*k/4
    
    g=np.array([K[ii] for ii in range(3) for jj in range(4)])
                
    return g

@nb.njit()
def getAdmittanceIndices():
    nodesA=np.array([0,2,4,6,0,1,4,5,0,1,2,3],dtype=np.int64)
    offsets=np.array([2**np.floor(ii/4) for ii in range(12)],dtype=np.int64)
    
    # edges=np.array([[self.globalNodeIndices[a],self.globalNodeIndices[a+o]] for a,o in zip(nodesA,offsets)])
    # edges=np.empty((12,2),dtype=np.int64)
    # for ii in range(12):
    #     nodeA=nodesA[ii]
    #     nodeB=nodeA+offsets[ii]
    #     edges[ii,0]=self.globalNodeIndices[nodeA]
    #     edges[ii,1]=self.globalNodeIndices[nodeB]
    
    edges=np.vstack((nodesA,nodesA+offsets)).transpose()
    return edges