#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:15:04 2021
Visualization routines for meshes
@author: benoit
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interp2d

import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


import pandas
from util import uniformResample,edgeRoles

def plotBoundEffect(fname,ycat='FVU',xcat='Domain size',groupby='Number of elements',legstem='%d elements'):
    dframe,_=importRunset(fname)
    dX=dframe[xcat]
    dEl=dframe[groupby]
    yval=dframe[ycat]
    l0=np.array([x/(n**(1/3)) for x,n in zip(dX,dEl)])
    
    sizes=np.unique(dEl)
    nSizes=len(sizes)
    labs=[legstem%n for n in sizes]
    
    
    def makePlot(xvals,xlabel):
        plt.figure()
        ax=plt.gca()
        ax.set_title('Simulation accuracy, uniform mesh')
        ax.grid(True)
        ax.set_ylabel('FVU')
        ax.set_xlabel(xlabel)
        outsideLegend()
        for ii in range(nSizes):
            sel=dEl==sizes[ii]
            plt.loglog(xvals[sel],yval[sel],marker='.',label=labs[ii])
        
        outsideLegend()
        plt.tight_layout()
        
    makePlot(dX,'Domain size [m]')
    makePlot(l0,'Element $l_0$ [m]')
    
    
def groupedScatter(fname,xcat,ycat,groupcat,filtercat=None,df=None):
    if df is None:
        df,cats=importRunset(fname)
    else:
        cats=df.keys()
    
    dX=df[xcat]
    dY=df[ycat]
    dL=df[groupcat]
    
    catNames=dL.unique()

    ax=plt.figure().add_subplot()
    ax.grid(True)
    ax.set_xlabel(xcat)
    ax.set_ylabel(ycat)
    # outsideLegend()
    for ii,label in enumerate(catNames):
        sel=dL==label
        plt.loglog(dX[sel],dY[sel],marker='.',
                   label=groupcat+': '+str(label))
        
    
    outsideLegend()
    plt.tight_layout()

def importRunset(fname):
    df=pandas.read_csv(fname)
    cats=df.keys()
    
    return df,cats

def importAndPlotTimes(fname,onlyCat=None,onlyVal=None,xCat='Number of elements'):
    df, cats = importRunset(fname)
    if onlyVal is not None:
        df=df[df[onlyCat]==onlyVal]
    
    
    xvals=df[xCat].to_numpy()
    nSkip=1+np.argwhere(cats=='Total time')[0][0]
    
    def getTimes(df,cats):
        cols=cats[6:nSkip]
        return df[cols].to_numpy().transpose()
    
    stepTimes=getTimes(df,cats)
    stepNames=cats[6:nSkip]
    
    ax=plt.figure().add_subplot()
    
    stackedTimePlot(ax, xvals, stepTimes, stepNames)
    ax.set_xlabel(xCat)
    ax.set_ylabel("Execution time [s]")
    ax.figure.tight_layout()

def stackedTimePlot(axis,xvals,stepTimes,stepNames):
    axis.stackplot(xvals,stepTimes,baseline='zero',labels=stepNames)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # axis.figure.tight_layout()
    axis.xaxis.set_major_formatter(mpl.ticker.EngFormatter())



def outsideLegend(**kwargs):
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,**kwargs)
    

def showEdges(axis,coords, edgeIndices, edgeVals=None,colorbar=True):
    edgePts=[[coords[a,:],coords[b,:]] for a,b in edgeIndices]
    
    if edgeVals is not None:
        (cMap, cNorm) = getCmap(edgeVals)
        gColor = cMap(cNorm(edgeVals))
        alpha=1.
        if colorbar:
            axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))
    
    else:
        # gColor=np.repeat([0,0,0,1], edgeIndices.shape[0])
        gColor=(0.,0.,0.)
        alpha=0.05
    
    gCollection=p3d.art3d.Line3DCollection(edgePts,colors=gColor,alpha=alpha)
    
    axis.add_collection(gCollection)


    
def showNodes(axis, coords, nodeVals):
    
    (cMap, cNorm) = getCmap(nodeVals)
    vColors = cMap(cNorm(nodeVals))
    x,y,z=np.hsplit(coords,3)
    
    axis.scatter3D(x,y,z, c=vColors)
    axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))


def getCmap(vals,forceBipolar=False):
    mn=min(vals)
    mx=max(vals)
    
    if ((mn < 0) and (abs(mn)/mx>0.01)) or forceBipolar:
        cMap = mpl.cm.seismic
        cNorm = mpl.colors.CenteredNorm()
    else:
        cMap = mpl.cm.plasma
        cNorm = mpl.colors.Normalize(vmin=0,vmax=mx)
    return (cMap, cNorm)


def new3dPlot(boundingBox):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    fig=plt.figure()
    axis=fig.add_subplot(projection='3d')
    origin=boundingBox[:3]
    span=boundingBox[3:]-origin
    center=0.5*(origin+boundingBox[3:])
    
    max_range=max(span)    
    
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*center[0]
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*center[1]
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*center[2]
    for xb, yb, zb in zip(Xb, Yb, Zb):
       axis.plot([xb], [yb], [zb], color=[1,1,1,0])

    return axis

def equalizeScale(ax):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    xy=ax.xy_dataLim
    zz=ax.zz_dataLim
    
    max_range = np.array([xy.xmax-xy.xmin,xy.ymax-xy.ymin, zz.max.max()-zz.min.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xy.xmax+xy.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(xy.ymax+xy.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zz.max.max()+zz.min.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], color=[1,1,1,0])

def showRawSlice(valList,ndiv):
    nX=ndiv+1
    nslice=nX//2
    
    
    vMat=valList.reshape((nX,nX,nX))
    
    vSelection=vMat[:,:,nslice].squeeze()
    
    (cMap, cNorm) = getCmap(vSelection.ravel())
    
    
    plt.imshow(vSelection, cmap=cMap, norm=cNorm)
    plt.colorbar()
    
def showCurrentVecs(axis, pts,vecs):
    X,Y,Z=np.hsplit(pts,3)
    dx,dy,dz=np.hsplit(vecs,3)
    
    iMag=np.linalg.norm(vecs,axis=1)
    
    cMap,cNorm=getCmap(iMag)
    
    colors=cMap(iMag)
    
    axis.quiver3D(X,Y,Z,dx,dy,dz, colors=colors)
    
 #TODO: deprecate   
def showSlice(coords,vals,nX=100,sliceCoord=0,axis=2,plotWhich=None,edges=None,edgeColors=None,forceBipolar=False,axes=None,xmax=None):
    axNames=[c for c in 'XYZ' if c!=axis]
    if plotWhich is None:
        plotWhich=True

    if xmax is None:
        xx0,yy0=np.hsplit(sliceCoords,2)
        bnds=np.array([min(xx0), max(xx0), min(yy0), max(yy0)]).squeeze()
    else:
        bnds=np.array([-xmax,xmax,-xmax,xmax])
        
    
    vImg=vals
    
    (cMap, cNorm) = getCmap(vImg.ravel(),forceBipolar)
    
    
    if axes is None:
        ax=plt.gca()
        cax=None
    else:
        ax=axes[0]
        cax=axes[1]

    
    img=ax.imshow(vImg, origin='lower', extent=bnds,
               cmap=cMap, norm=cNorm,interpolation='bilinear')
    # cbar=plt.colorbar()
    cbar=plt.gcf().colorbar(img,ax=ax,cax=cax)
    
    
    engform=mpl.ticker.EngFormatter()
    ax.xaxis.set_major_formatter(engform)
    ax.yaxis.set_major_formatter(engform)
    ax.set_xlabel(axNames[0]+ ' [m]')
    ax.set_ylabel(axNames[1]+' [m]')
    
    return [img]

def showEdges2d(axis,edgePoints,edgeColors=None,**kwargs):
    if edgeColors is None:
        edgeColors=(0.,0.,0.,)
        
        kwargs['alpha']=0.05
        # alpha=0.05
            
    edgeCol=mpl.collections.LineCollection(edgePoints, colors=edgeColors,
                                           linewidths=0.5, **kwargs)
    axis.add_collection(edgeCol)
    return edgeCol

def centerSlice(fig,simulation,grid=None):
    artists=[]
    isNodeInPlane=simulation.mesh.nodeCoords[:,-1]==0
    intcoord=simulation.intifyCoords()
    pcoord=intcoord[isNodeInPlane,:-1]
    xyCoord=simulation.mesh.nodeCoords[:,:-1]
    ccoord=xyCoord[isNodeInPlane]
    pv=simulation.nodeVoltages[isNodeInPlane]
    rElec=simulation.currentSources[0].radius

    nx=simulation.ptPerAxis
    if nx>2**12:
        nx=2**10+1
        
    

    xmax=max(ccoord[:,1])
    xx=np.linspace(min(ccoord[:,0]),
                   xmax,
                   nx)
    XX,YY=np.meshgrid(xx,xx)
    
    xx0,yy0=np.hsplit(xyCoord,2)
    bnds=np.array([min(xx0), max(xx0), min(yy0), max(yy0)]).squeeze()
    bndInset=rElec*np.array([-3., 3., -3., 3.])
    zoomFactor=np.mean(0.4*bnds/bndInset)
    
    r=np.sqrt(XX**2 + YY**2)
    
    
    triSet=tri.Triangulation(ccoord[:,0], ccoord[:,1])    
    interp=tri.LinearTriInterpolator(triSet, pv)
    vInterp=interp(XX.ravel(),YY.ravel()).reshape((nx,nx))
    
    
    
    inside=r<rElec
    ana=np.empty_like(r)
    ana[inside]=1
    ana[~inside]=rElec/r[~inside]
    
    err=vInterp-ana
    
    interpCoords=np.vstack((XX.ravel(), YY.ravel())).transpose()

    filteredRoles=simulation.nodeRoleTable.copy()
    filteredRoles[~isNodeInPlane]=-1
    allEdges=np.array(simulation.edges)
    roles=edgeRoles(allEdges,filteredRoles)
    
    isEdgeInPlane=~np.any(roles==-1,axis=1)
    touchesSource=roles==2
    
    isInvalid=np.all(touchesSource,axis=1)|(~isEdgeInPlane)
    isBoundaryNode=np.logical_and(touchesSource,np.sum(touchesSource,axis=1,keepdims=True)==1)
    
    boundaryNodes=allEdges[isBoundaryNode&np.expand_dims(isEdgeInPlane,axis=1)]
    
    edgePoints=toEdgePoints(xyCoord,allEdges[~isInvalid])
    sourceEdge=[[np.zeros(2),xy] for xy in xyCoord[boundaryNodes]]
        
        
    if fig.axes==[]:    
        grid = AxesGrid(fig, 111,  # similar to subplot(144)
                    nrows_ncols=(1, 2),
                    axes_pad=(0.45, 0.15),
                    # axes_pad=0.2,
                    label_mode="L",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="5%",
                    cbar_pad="2%",
                    )
    
   
        grid[0].set_title('Simulated potential [V]')
        grid[1].set_title('Absolute error [V]')
        # axV=fig.add_subplot(1,2,1)
        # axE=fig.add_subplot(1,2,2)
        # caxV=None
        # caxE=None
        # axV.set_title('Simulated potential [V]')
        # axE.set_title('Absolute error [V]')
        
        engform=mpl.ticker.EngFormatter()
        # for ax in [axV,axE]:
        for ax in grid:
            ax.xaxis.set_major_formatter(engform)
            ax.yaxis.set_major_formatter(engform)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_xlim(min(ccoord[:,0]),max(ccoord[:,0]))
            ax.set_ylim(min(ccoord[:,1]),max(ccoord[:,1]))
           
        vsub=zoomed_inset_axes(grid[0],zoom=zoomFactor, loc=1)
        vsub.set_xlim(bndInset[0],bndInset[1])
        vsub.set_ylim(bndInset[2],bndInset[3])
        mark_inset(grid[0],vsub,loc1=2, loc2=4,
                   fc='none', ec="0.5")
    
            # ax.set_xlim(-3e-6, 3e-6)
            # ax.set_ylim(-3e-6, 3e-6)
    # else:
        # axV=fig.axes[0]
        # axE=fig.axes[1]
        # caxV=fig.axes[2]
        # caxE=fig.axes[3]
    # vmin=min(vInterp.ravel())
    # vmax=max(vInterp.ravel())
    # ermax=max(err.ravel())
    
    
    
    
    vmap,vnorm=getCmap(vInterp.ravel())
    emap,enorm=getCmap(err.ravel(),forceBipolar=True)


    
    vArt=grid[0].imshow(vInterp, origin='lower', extent=bnds,
               # cmap=vmap, norm=vnorm,interpolation='bilinear')
               cmap=vmap, interpolation='bilinear')
    grid.cbar_axes[0].colorbar(vArt)
    
    # artists.append(vsub.imshow(vInterp, origin='lower', extent=bnds,
    #             cmap=vmap, norm=vnorm,interpolation='bilinear'))
    
    
    errArt=grid[1].imshow(err, origin='lower', extent=bnds,
                cmap=emap, norm=enorm,interpolation='bilinear')
               # cmap=emap, interpolation='bilinear')
    grid.cbar_axes[1].colorbar(errArt)
    
    # vArt=showSlice(interpCoords, vInterp, nX=nx,axes=[axV,caxV],xmax=xmax)
    # vArt.append(showEdges2d(axV, edgePoints))
    # errArt=showSlice(interpCoords,err, forceBipolar=True,nX=nx,axes=[axE,caxE],xmax=xmax)
    # errArt.append(showEdges2d(axE, edgePoints))
    
    # for ax, art in zip([axV,axE],[vArt,errArt]):
    # axlist=[vsub]
    # axlist.extend(grid)
    # for ax in axlist:
    for ax in grid:
        artists.append(showEdges2d(ax, edgePoints))
        artists.append(showEdges2d(ax, sourceEdge,edgeColors=(.5,.5,.5),alpha=.25,linestyles=':'))
        
        tht=np.linspace(0,2*np.pi)
        x=rElec*np.cos(tht)
        y=rElec*np.sin(tht)
        artists.append(ax.plot(x,y,'k:',alpha=0.5)[0])
    
    # vArt=patchworkImage(axV, caxV, qc, qv)
    # vArt.append(showEdges2d(axV, edgePoints))
    # errArt=patchworkImage(axE, caxE, qc, qerr)
    # errArt.append(showEdges2d(axE, edgePoints))
    
    artists.append(errArt)
    artists.append(vArt)
    # plt.tight_layout()
    
    return artists,grid

def patchworkImage(axis,caxis,quadBbox,quadVal):
    imlist=[]
    cMap,cNorm=getCmap(quadVal.ravel())
    
    for c,v in zip(quadVal,quadBbox):
        im=axis.imshow(v.reshape((2,2)),
               origin='lower',extent=c.squeeze(),
                cmap=cMap, norm=cNorm,interpolation='bilinear')
        imlist.append(im)
        
    cbar=plt.gcf().colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap),
                            ax=axis,cax=caxis) 
    
    return imlist,grid

def getPlanarEdgePoints(coords,edges, normalAxis=2, axisCoord=0):

    sel=np.equal(coords[:,normalAxis],axisCoord)
    selAx=np.array([n for n in range(3) if n!=normalAxis])
        
    planeCoords=coords[:,selAx]
        
    edgePts=[[planeCoords[a,:],planeCoords[b,:]] for a,b in edges if sel[a]&sel[b]]

    return edgePts

def toEdgePoints(planeCoords,edges):
    edgePts=np.array([planeCoords[e] for e in edges])
    
    return edgePts
    
def error2d(fig,simulation,rElec=1e-6,datalabel='Simulation'):
    
    # # nTypes,_,_,_,_=simulation.getNodeTypes()
    # # rest=np.logical_or(nTypes==0,nTypes==2)
    isDoF=simulation.nodeRoleTable==0
    v=simulation.nodeVoltages
    
    nNodes=len(simulation.mesh.nodeCoords)
    nElems=len(simulation.mesh.elements)
    r=np.linalg.norm(simulation.mesh.nodeCoords,axis=1)
    
    
    
    sorter=np.argsort(r)
    rsort=r[sorter]
    vsort=v[sorter]
    rsort[0]=rElec/2

    # # rDense=np.linspace(0,max(r[rest]),100)
    # # rmin=min(rElec/2,min(rsort[1:]))
    lmin=np.log10(rElec/2)
    lmax=np.log10(max(r))
    rDense=np.logspace(lmin,lmax,200)
    
    def vAna(r):
        r0=r.copy()
        r0[r<rElec]=rElec
        return rElec/(r0)
    
    # analytic=vAna(rsort)
    analyticDense=vAna(rDense)
        
    # # FVU=xCell.getFVU(v,analytic,rest)
    # err=vsort-analytic
    # FVU=np.trapz(np.abs(err),rsort)/np.trapz(analyticDense,rDense)
    
    # # errRel=err/analytic
    # # fig2d, axes=plt.subplots(2,1)
    # # ax2dA,ax2dB=axes
    
    err,FVU = simulation.calculateErrors()
    
    # filter non DoF
    filt=isDoF[sorter]
    
    rFilt=rsort[filt]
    errFilt=err[filt]
    
    
    if fig.axes==[]:
        ax2dA=fig.add_subplot(2,1,1)
        ax2dB=fig.add_subplot(2,1,2)
        ax2dA.grid(True)
        ax2dB.grid(True)
        ax2dA.set_xlim(left=rElec/2,right=2*max(r))
            
        ax2dA.xaxis.set_major_formatter(mpl.ticker.EngFormatter())  
        ax2dA.set_xlabel('Distance from source [m]')
        ax2dA.set_ylabel('Voltage [V]')
        ax2dB.set_ylabel('Absolute error [V]')

        ax2dA.set_xscale('log')
        ax2dB.set_xscale('log')
        ax2dB.sharex(ax2dA)
        # ax2dB.autoscale(enable=True, axis='y', tight=True)
        
        anaLine=ax2dA.plot(rDense,analyticDense,c='b',label='Analytical')

        isNew=True
    else:
        ax2dA,ax2dB=fig.axes
        isNew=False
        # ax2dA.cla()
        # ax2dB.cla()
        
 
    simLine=ax2dA.plot(rsort,vsort,c='k',marker='.',label=datalabel)
    if isNew:
        ax2dA.legend()
    # title=fig.suptitle('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
    title=fig.text(0.5,1.01,
                   '%s:%d nodes, %d elements, error %.2g Vm'%(simulation.meshtype,nNodes,nElems,FVU),
                    horizontalalignment='center', verticalalignment='bottom', transform=ax2dA.transAxes)

    errLine=ax2dB.plot(rFilt,errFilt,c='r',marker='.',linestyle='None',label=datalabel)
    # errLine=ax2dB.plot(rsort,err,c='r',marker='.',label=datalabel)

    plt.tight_layout()
    return [simLine[0],errLine[0],title]
  
    
class FigureAnimator:
    def __init__(self,fig,study):
        pass
    def addSimulationData(self,sim):
        pass
    def getArtists(self):
        pass
    
class SliceSet(FigureAnimator):
    def __init__(self,fig,study):
        self.bnds=study.bbox[[0,3,2,4]]
        self.grid = AxesGrid(fig, 211,  # similar to subplot(144)
                    nrows_ncols=(1, 2),
                    axes_pad=(0.45, .15),
                    # axes_pad=0.2,
                    label_mode="L",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="5%",
                    cbar_pad="2%",
                    )
        
        
    
   
        self.grid[0].set_title('Simulated potential [V]')
        self.grid[1].set_title('Absolute error [V]')

        
        engform=mpl.ticker.EngFormatter()
        # for ax in [axV,axE]:
        for ax in self.grid[:2]:
            ax.xaxis.set_major_formatter(engform)
            ax.yaxis.set_major_formatter(engform)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_xlim(self.bnds[0],self.bnds[1])
            ax.set_ylim(self.bnds[2],self.bnds[3])
            
        self.vArrays=[]
        self.errArrays=[]
        
        self.vbounds=np.zeros(2)
        self.errbounds=np.zeros(2)
        
        self.meshEdges=[]
        self.sourceEdges=[]
        
        self.rElec=0
        
           
        
        
    # def addInset(self,axIndex):
    #     vsub=zoomed_inset_axes(grid[0],zoom=zoomFactor, loc=1)
    #     vsub.set_xlim(bndInset[0],bndInset[1])
    #     vsub.set_ylim(bndInset[2],bndInset[3])
    #     mark_inset(grid[0],vsub,loc1=2, loc2=4,
    #                fc='none', ec="0.5")
    
    def addSimulationData(self,sim):
        isNodeInPlane=sim.mesh.nodeCoords[:,-1]==0
        intcoord=sim.intifyCoords()
        pcoord=intcoord[isNodeInPlane,:-1]
        xyCoord=sim.mesh.nodeCoords[:,:-1]
        ccoord=xyCoord[isNodeInPlane]
        pv=sim.nodeVoltages[isNodeInPlane]
        rElec=sim.currentSources[0].radius
        self.rElec=rElec
        

        nx=sim.ptPerAxis
        if nx>2**12:
            nx=2**10+1
            
        
    
        xmax=max(ccoord[:,1])
        xx=np.linspace(min(ccoord[:,0]),
                       xmax,
                       nx)
        XX,YY=np.meshgrid(xx,xx)
        
        xx0,yy0=np.hsplit(xyCoord,2)
        bnds=np.array([min(xx0), max(xx0), min(yy0), max(yy0)]).squeeze()
        bndInset=rElec*np.array([-3., 3., -3., 3.])
        zoomFactor=np.mean(0.4*bnds/bndInset)
        
        r=np.sqrt(XX**2 + YY**2)
        
        
        triSet=tri.Triangulation(ccoord[:,0], ccoord[:,1])    
        interp=tri.LinearTriInterpolator(triSet, pv)
        vInterp=interp(XX.ravel(),YY.ravel()).reshape((nx,nx))
        
        
        
        inside=r<rElec
        ana=np.empty_like(r)
        ana[inside]=1
        ana[~inside]=rElec/r[~inside]
        
        err=vInterp-ana
        
        self.vbounds[0]=min(self.vbounds[0],min(vInterp.ravel()))
        self.vbounds[1]=max(self.vbounds[1],max(vInterp.ravel()))
        
        self.errbounds[0]=min(self.errbounds[0],min(err.ravel()))
        self.errbounds[1]=max(self.errbounds[1],max(err.ravel()))

        
        interpCoords=np.vstack((XX.ravel(), YY.ravel())).transpose()
    
        filteredRoles=sim.nodeRoleTable.copy()
        filteredRoles[~isNodeInPlane]=-1
        allEdges=np.array(sim.edges)
        roles=edgeRoles(allEdges,filteredRoles)
        
        isEdgeInPlane=~np.any(roles==-1,axis=1)
        touchesSource=roles==2
        
        isInvalid=np.all(touchesSource,axis=1)|(~isEdgeInPlane)
        isBoundaryNode=np.logical_and(touchesSource,np.sum(touchesSource,axis=1,keepdims=True)==1)
        
        boundaryNodes=allEdges[isBoundaryNode&np.expand_dims(isEdgeInPlane,axis=1)]
        
        edgePoints=toEdgePoints(xyCoord,allEdges[~isInvalid])
        sourceEdge=[[np.zeros(2),xy] for xy in xyCoord[boundaryNodes]]
          
        
        self.vArrays.append(vInterp)
        self.errArrays.append(err)
        self.meshEdges.append(edgePoints)
        self.sourceEdges.append(sourceEdge)
        
    def getArtists(self):
        vmap,vnorm=getCmap(self.vbounds)
        emap,enorm=getCmap(self.errbounds, forceBipolar=True)
        artistSet=[]
        
        if self.rElec>0:
            tht=np.linspace(0,2*np.pi)
            x=self.rElec*np.cos(tht)
            y=self.rElec*np.sin(tht)
            
            rInset=3*self.rElec
            insetZoom=0.8*self.bnds[1]/rInset
            
            for ax in self.grid:
                ax.plot(x,y,'k',alpha=0.5)
                
        if insetZoom>0:
            inset=[]
            for ii in range(2):
                
                inset.append(zoomed_inset_axes(self.grid[ii],zoom=insetZoom, 
                                               # loc=1))
                                               bbox_to_anchor=(.5, -.8), loc='center',
                                               bbox_transform=self.grid[ii].transAxes))
                inset[ii].set_xlim(-rInset,rInset)
                inset[ii].set_ylim(-rInset,rInset)
                # inset[ii].set_xlim(,rInset)
                # inset[ii].set_ylim(-rInset,0)
                mark_inset(self.grid[ii],inset[ii],loc1=1, loc2=2,
                           fc='none', ec="0.5")
                
                # inset[ii].yaxis.get_major_locator().set_params(nbins=3)
                # inset[ii].xaxis.get_major_locator().set_params(nbins=3)
                # inset[ii].tick_params(labelleft=False, labelbottom=False)
            
            for ax in inset:
                ax.plot(x,y,'k',alpha=0.5)

        for ii in range(len(self.vArrays)):
            artists=[]
            vInterp=self.vArrays[ii]
            err=self.errArrays[ii]
            edgePoints=self.meshEdges[ii]
            sourceEdge=self.sourceEdges[ii]
            vArt=self.grid[0].imshow(vInterp, origin='lower', extent=self.bnds,
                        cmap=vmap, norm=vnorm,interpolation='bilinear')
            self.grid.cbar_axes[0].colorbar(vArt)
            
                    
            errArt=self.grid[1].imshow(err, origin='lower', extent=self.bnds,
                        cmap=emap, norm=enorm,interpolation='bilinear')
            self.grid.cbar_axes[1].colorbar(errArt)
            
            
            if insetZoom>0:
                artists.append(inset[0].imshow(vInterp, origin='lower', extent=self.bnds,
                            cmap=vmap, norm=vnorm,interpolation='bilinear'))
                artists.append(inset[1].imshow(err, origin='lower', extent=self.bnds,
                            cmap=emap, norm=enorm,interpolation='bilinear'))
            
    

            
            for ax in self.grid:
                artists.append(showEdges2d(ax, edgePoints))
                
            for ax in inset:
                artists.append(showEdges2d(ax, edgePoints))
                artists.append(showEdges2d(ax, sourceEdge,edgeColors=(.5,.5,.5),alpha=.25,linestyles=':'))
                
                
            
            artists.append(errArt)
            artists.append(vArt)
            artistSet.append(artists)
        
        return artistSet
    
    
class ErrorGraph(FigureAnimator):
    def __init__(self,fig,study):
        self.fig=fig
        axV=fig.add_subplot(2,1,1)
        axErr=fig.add_subplot(2,1,2)
        axV.grid(True)
        axErr.grid(True)
        
            
        axV.xaxis.set_major_formatter(mpl.ticker.EngFormatter())  
        axV.set_xlabel('Distance from source [m]')
        axV.set_ylabel('Voltage [V]')
        axErr.set_ylabel('Absolute error [V]')

        axV.set_xscale('log')
        axErr.set_xscale('log')
        axErr.sharex(axV)
        
        self.axV=axV
        self.manualBounds=False
        
        self.axErr=axErr
    
        self.titles=[]
        self.errR=[]
        self.errors=[]
        self.simR=[]
        self.sims=[]
        
        self.analytic=[]
        self.analyticR=[]

    
    def addSimulationData(self, sim):
        isDoF=sim.nodeRoleTable==0
        v=sim.nodeVoltages
        
        nNodes=len(sim.mesh.nodeCoords)
        nElems=len(sim.mesh.elements)
        r=np.linalg.norm(sim.mesh.nodeCoords,axis=1)
        
        rElec=sim.currentSources[0].radius
        
        sorter=np.argsort(r)
        rsort=r[sorter]
        vsort=v[sorter]
        rsort[0]=rElec/2
    
        lmin=np.log10(rElec/2)
        lmax=np.log10(max(r))
        
        if len(self.analyticR)==0:
            rDense=np.logspace(lmin,lmax,200)
        
            def vAna(r):
                r0=r.copy()
                r0[r<rElec]=rElec
                return rElec/(r0)
            
            # analytic=vAna(rsort)
            analyticDense=vAna(rDense)
            self.analyticR=rDense
            self.analytic=analyticDense
            
            
            self.axV.set_xlim(left=rElec/2,right=2*max(r))
            
            
        err,FVU = sim.calculateErrors()
        
        # filter non DoF
        filt=isDoF[sorter]
        
        rFilt=rsort[filt]
        errFilt=err[filt]
        
        self.titles.append('%s:%d nodes, %d elements, error %.2g Vm'%(sim.meshtype,nNodes,nElems,FVU),)
        self.errR.append(rFilt)
        self.errors.append(errFilt)
        self.simR.append(rsort)
        self.sims.append(vsort)
        
        
    def getArtists(self):
        artistSet=[]
        
        self.axV.plot(self.analyticR,self.analytic,c='b',label='Analytical')
     
        for ii in range(len(self.titles)):
            artists=[]
            simLine=self.axV.plot(self.simR[ii],self.sims[ii],
                                  c='k',marker='.',label='Simulated')
            if ii==0:
                self.axV.legend()
            # title=fig.suptitle('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
            title=self.fig.text(0.5,1.01,
                           self.titles[ii],
                            horizontalalignment='center', 
                            verticalalignment='bottom', 
                            transform=self.axV.transAxes)
        
            errLine=self.axErr.plot(self.errR[ii],self.errors[ii],
                               c='r',marker='.',linestyle='None')
            # errLine=axErr.plot(rsort,err,c='r',marker='.',label=datalabel)
        
            plt.tight_layout()
            artists.append(title)
            artists.append(errLine[0])
            artists.append(simLine[0])
            
            artistSet.append(artists)
        return artistSet
        
        
        