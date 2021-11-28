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

import matplotlib.tri as tri

import pandas

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

def importRunset(fname):
    df=pandas.read_csv(fname)
    cats=df.keys()
    
    return df,cats

def importAndPlotTimes(fname,onlyType=None,xCat='Number of elements',numExtraFields=0):
    df, cats = importRunset(fname)
    if onlyType is not None:
        df=df[df['Element type']==onlyType]
    
    
    xvals=df[xCat].to_numpy()
    nSkip=-1-numExtraFields
    
    def getTimes(df,cats):
        cols=cats[5:nSkip]
        return df[cols].to_numpy().transpose()
    
    stepTimes=getTimes(df,cats)
    stepNames=cats[5:nSkip]
    
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
    else:
        # gColor=np.repeat([0,0,0,1], edgeIndices.shape[0])
        gColor=(0.,0.,0.)
        alpha=0.05
    
    gCollection=p3d.art3d.Line3DCollection(edgePts,colors=gColor,alpha=alpha)
    
    axis.add_collection(gCollection)

    if colorbar:
        axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))
    
    
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
        cNorm = mpl.colors.Normalize()
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
    
    
def showSlice(coords,vals,nX=100,sliceCoord=0,axis=2,plotWhich=None,edges=None,edgeColors=None,forceBipolar=False):
    axNames=[c for c in 'XYZ' if c!=axis]
    
    if plotWhich is None:
        plotWhich=True
    
    sel=np.logical_and(coords[:,axis]==sliceCoord, plotWhich)
    selAx=np.array([n for n in range(3) if n!=axis])
    
    planeCoords=coords[:,selAx]
    sliceCoords=planeCoords[sel]
    sliceVals=vals[sel]
    
    xx0,yy0=np.hsplit(sliceCoords,2)
    bnds=np.array([min(xx0), max(xx0), min(yy0), max(yy0)]).squeeze()
    
    
    
    xx=np.linspace(bnds[0],bnds[1],nX)
    yy=np.linspace(bnds[2],bnds[3],nX)
    XX,YY=np.meshgrid(xx,yy)
    
    

    triang = tri.Triangulation(xx0.squeeze(), yy0.squeeze())
    interpolator = tri.LinearTriInterpolator(triang, sliceVals)
    vImg= interpolator(XX, YY)
    
    (cMap, cNorm) = getCmap(vImg.ravel(),forceBipolar)
    
    img=plt.imshow(vImg, origin='lower', extent=bnds,
               cmap=cMap, norm=cNorm,interpolation='bilinear')
    plt.colorbar()
    
    ax=plt.gca()
    engform=mpl.ticker.EngFormatter()
    ax.xaxis.set_major_formatter(engform)
    ax.yaxis.set_major_formatter(engform)
    plt.xlabel(axNames[0]+ ' [m]')
    plt.ylabel(axNames[1]+' [m]')
    
    if edges is not None:
        if edgeColors is None:
            edgeColors=(0.,0.,0.,)
            alpha=0.05
            
        edgePts=[[planeCoords[a,:],planeCoords[b,:]] for a,b in edges]
        
        edgeCol=mpl.collections.LineCollection(edgePts, colors=edgeColors,alpha=0.05,
                                               linewidths=0.5)
        plt.gca().add_collection(edgeCol)


    return img


def centerSlice(fig,simulation):
    
    coords=simulation.mesh.nodeCoords
    v=simulation.nodeVoltages
    nTypes,_,_,_,_=simulation.getNodeTypes()
    rest=nTypes==0
    edges=simulation.ed
    
    r=np.linalg.norm(simulation.mesh.nodeCoords,axis=1)
    rDense=np.linspace(min(r[rest]),max(r[rest]),100)
    
    analytic=analyticVsrc(np.zeros(3), srcAmplitude, r,srcType=srcType)
    analyticDense=analyticVsrc(np.zeros(3), srcAmplitude, rDense,srcType=srcType)
        
        
    
    showSlice(coords, vals, plotWhich=rest, edges=edges)
    
    
    
    
def error2d(fig,simulation):
    
    nTypes,_,_,_,_=simulation.getNodeTypes()
    rest=nTypes==0
    v=simulation.nodeVoltages
    
    nNodes=len(simulation.mesh.nodeCoords)
    nElems=len(simulation.mesh.elements)
    r=np.linalg.norm(simulation.mesh.nodeCoords,axis=1)
    rDense=np.linspace(0,max(r[rest]),100)
    
    def vAna(r):
        r[r<1e-6]=1e-6
        return 1e-6/(4*np.pi*r)
    
    analytic=vAna(r)
    analyticDense=vAna(rDense)
        
    # FVU=xCell.getFVU(v,analytic,rest)
    err=v-analytic
    FVU=np.trapz(err,r)
    
    # fig2d, axes=plt.subplots(2,1)
    # ax2dA,ax2dB=axes
    
    if fig.axes==[]:
        ax2dA=fig.add_subplot(2,1,1)
        ax2dB=fig.add_subplot(2,1,2)
            
        ax2dA.xaxis.set_major_formatter(mpl.ticker.EngFormatter())  
        ax2dA.set_xlabel('Distance from source [m]')
        ax2dA.set_ylabel('Voltage [V]')
        ax2dB.set_ylabel('Absolute error [V]')
        ax2dB.sharex(ax2dA)
    
        isNew=True
    else:
        ax2dA,ax2dB=fig.axes
        isNew=False
        # ax2dA.cla()
        # ax2dB.cla()
        
 
    anaLine=ax2dA.plot(rDense,analyticDense,c='b', label='Analytical')
    simLine=ax2dA.scatter(r[rest],v[rest],c='k',label='Simulation')
    if isNew:
        ax2dA.legend()
    # title=fig.suptitle('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
    title=fig.text(0.5,1.01,'%d nodes, %d elements, error %.2g Vm'%(nNodes,nElems,FVU),
                    horizontalalignment='center', verticalalignment='bottom', transform=ax2dA.transAxes)

    errLine=ax2dB.scatter(r[rest],err[rest],c='r',label='Absolute')

    plt.tight_layout()
    return [anaLine[0],simLine,errLine,title]
    
    # if savePlots:
    #     figResult.savefig(self.resultPath+'result_'+self.iteration)
    #     figImage.savefig(self.resultPath+'errorImage_'+self.iteration)
    #     fig2d.savefig(self.resultPath+'errorPlot_'+self.iteration)


class ErrorGraph():
    def __init__(self,figure):
        self.fig=figure
        
        if figure.axes==[]:
            #not yet setup
            ax1=figure.add_subplot(2,1,1)
            ax2=figure.add_subplot(2,1,2)
            
            ax1.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
            anaLine=ax1.plot(0,0,label='Analytical')
            simLine=ax1.plot(0,0,c='k',linestyle=None,label='Simulation',marker='.')
            ax1.set_xlabel('Distance from source [m]')
            ax1.set_ylabel('Voltage [V]')
            ax1.set_title('')
            ax1.legend()
            
            errLine=ax2.plot(0,0,c='r',linestyle=None,marker='.')
            ax2.set_ylabel('Absolute error [V]')
            ax2.sharex(ax1)


        else:
            ax1,ax2=figure.axes
    
        self.ax1=ax1
        self.ax2=ax2
        self.anaLine=anaLine[0]
        self.simLine=simLine[0]
        self.errLine=errLine[0]
            
    def update(self,simulation):
        nTypes,_,_,_,_=simulation.getNodeTypes()
        rest=nTypes==0
        v=simulation.nodeVoltages
        nNodes=len(simulation.mesh.nodeCoords)
        nElems=len(simulation.mesh.elements)
        r=np.linalg.norm(simulation.mesh.nodeCoords,axis=1)
        rDense=np.linspace(0,max(r[rest]),100)
        
        def vAna(r):
            r[r<1e-6]=1e-6
            return 1e-6/(4*np.pi*r)
        
        analytic=vAna(r)
        analyticDense=vAna(rDense)
            
        # FVU=xCell.getFVU(v,analytic,rest)
        err=v-analytic
        FVU=np.trapz(err,r)
        
        
        
        self.anaLine.set_data(rDense,analyticDense)
        self.simLine.set_data(r[rest],v[rest])
        self.errLine.set_data(r[rest],err[rest])
            
        self.ax1.title.set_text('%d nodes, %d elements\nError= %g Vm'%(nNodes,nElems,FVU))
        plt.tight_layout()
        return [self.anaLine, self.simLine, self.errLine]