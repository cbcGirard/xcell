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
from matplotlib.animation import ArtistAnimation
import numpy as np
# from scipy.interpolate import interp2d
from scipy.sparse import tril
import os

# import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset


import pandas
# from util import uniformResample, edgeRoles, getquads, quadsToMaskedArrays, coords2MaskedArrays
import util

class DataPrefs:
    def __init__(self, name,
                 dataSource=None,
                 scaleStyle=None,
                 cmap=None,
                 norm=None,
                 colors=None):
        self.name=name
        self.dataSource=dataSource
        self.scaleStyle=scaleStyle
        self.cmap=cmap
        self.norm=norm
        self.colors=colors
        self.range=ScaleRange()
        self.scaleViz=None
        self.dataViz=None
        
    def updateRange(self,newVals):
        self.range.update(newVals)
        
    def applyPrefs(self,axis,data):
        self.updateRange(data)
        plotargs={}
        
        if self.scaleStyle=='legend':
            if self.colors is None:
                self.colors=discreteLegend(axis, data)
            plotargs['colors']=self.colors
        
        if self.scaleStyle=='colorbar':
            if self.cmap is None:
                cmap,cnorm=getCmap(data)
            else:
                cmap=self.cmap
                cnorm=self.norm
            plotargs['cmap']=cmap
            plotargs['norm']=cnorm
            
            # plt.colorbar(mpl.cm.ScalarMappable(**plotargs),
            #              ax=axis)
            
        return plotargs
        
class DisplayPrefs:
    def __init__(self,nodePrefs=None, 
                 edgePrefs=None,
                 imgPrefs=None):
        if nodePrefs is None:
            nodePrefs=DataPrefs('nodePrefs')
        if edgePrefs is None:
            edgePrefs=DataPrefs('edgePrefs')
        if imgPrefs is None:
            imgPrefs=DataPrefs('imgPrefs')
        
        self.nodePrefs=nodePrefs
        self.edgePrefs=edgePrefs
        self.imgPrefs=imgPrefs
    

class SliceViewer:
    def __init__(self,axis,sim,displayPrefs=None,topoType='mesh'):
        if axis is None:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot()
        else:
            self.ax=axis
            self.fig=axis.figure
            
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.sim=sim
        self.nLevels=0
        
        formatXYAxis(self.ax, bounds=None)
        self.topoType=topoType

        self.setPlane()
        # self.drawFns=[]
        # self.drawArgs=[]
        
        if displayPrefs is None:
            displayPrefs=DisplayPrefs()
            
        self.prefs=displayPrefs
        self.edgeData=None
        self.nodeData=None
        self.edgeArtist=None
        self.nodeArtist=None
        self.edgeScaleViz=None
        self.nodeScaleViz=None
        # self.edgeColors=[]
        # self.nodeColors=[]
        
        
        
    def changeData(self,newSim):
        self.sim=newSim
        
        self.setPlane(self.normAxis,self.planeZ)
        
        
    def setPlane(self,normalAxis=2, normCoord=0):
        coords,_=self.sim.getCoords(self.topoType)
        ic=self.sim.intifyCoords(coords)        
        self.intCoord=ic
        
        pointCart=np.zeros(3)
        pointCart[normalAxis]=normCoord
        
        levelIndex=self.sim.intifyCoords(pointCart)[normalAxis]
        self.lvlIndex=levelIndex
        # self.nLevels=np.unique(ic[:,normalAxis]).shape[0]
        self.nLevels=max(ic[:,normalAxis])
        
        self.normAxis=normalAxis
        self.__setmask()
        
  
    def movePlane(self,stepAmount):
        self.lvlIndex+=stepAmount
        self.__setmask()
        
    def __setmask(self):
        axNames=['x','y','z']
        axlist=np.arange(3)
        
        coords,_=self.sim.getCoords(self.topoType)
        otherAxes=axlist!=self.normAxis
      
        inPlane=self.intCoord[:,self.normAxis]==self.lvlIndex
        zval=coords[inPlane,self.normAxis][0]
        self.planeZ=zval
        self.nInPlane=inPlane
        
        self.xyCoords=coords[:,otherAxes]
        self.zCoords=coords[:,~otherAxes].squeeze()
        self.pCoords=self.xyCoords[inPlane]
        
        #graph formatting
        xyNames=[axNames[i]+" [m]" for i in axlist[otherAxes]]
        titleStr=axNames[self.normAxis]+"=%.2g"%zval
        self.ax.set_xlabel(xyNames[0])
        self.ax.set_ylabel(xyNames[1])
        self.ax.set_title(titleStr)
        
        bndInds=[d+3*ii for d in axlist[otherAxes] for ii in range(2)]
        bnds=self.sim.mesh.bbox[bndInds]
        self.ax.set_xlim(bnds[0],bnds[1])
        self.ax.set_ylim(bnds[2],bnds[3])
        
    def showEdges(self,connAngle=None,colors=None,**kwargs):
        edge=self.sim.getEdges(self.topoType)
        edgeMask=self.nInPlane[edge]
        edgeInPlane=np.all(edgeMask,axis=1)
        
        planeEdges=edge[edgeInPlane]
        
        edgePts=toEdgePoints(self.xyCoords, planeEdges)
        
        if colors is not None:
            colors=colors[edgeInPlane]
        
        art=showEdges2d(self.ax,
                    edgePts,
                          colors,
                          **kwargs)
        
        if connAngle is not None:
            edgeAdjacent=np.logical_xor(edgeMask[:,0],
                                        edgeMask[:,1])
            dz=np.array(self.zCoords-self.planeZ,
                        ndmin=2).transpose()
            
            skew=np.matmul(dz,
                           np.array([np.cos(connAngle),np.sin(connAngle)],
                                    ndmin=2))
            skewPoints=toEdgePoints(self.xyCoords+skew,
                                    self.sim.getEdges()[edgeAdjacent])
            showEdges2d(self.ax, skewPoints, colors, **kwargs)
            
        return art
        
    def showNodes(self,nodeVals,colors=None,**kwargs):
        
        x,y=np.hsplit(self.pCoords,2)
        
        if colors is not None:
            art=self.ax.scatter(x,y,
                       color=colors[self.nInPlane],
                       **kwargs)
        else:
            if nodeVals is not None:
                vals=nodeVals[self.nInPlane]
            else:
                vals='k'
            art=self.ax.scatter(x,y,c=vals,**kwargs)
            
        return art
            
    def __drawSet(self):
        #keep limits on redraw
        undrawAxis(self.ax)
        
        if self.prefs.edgePrefs.name is not None:
            plotprefs=self.prefs.edgePrefs.applyPrefs(
                self.ax, self.edgeData)
            self.edgeArtist=self.showEdges(**plotprefs)
            
        if self.prefs.nodePrefs.name is not None:
            plotprefs=self.prefs.nodePrefs.applyPrefs(
                self.ax, self.nodeData)
            self.nodeArtist=self.showNodes(self.nodeData,**plotprefs)
            
        self.fig.canvas.draw()

            
    def onKey(self,event):
        step=None
        if event.key=='up':
            if self.lvlIndex<self.nLevels:     
                step=1
        if event.key=='down':
            if self.lvlIndex>0:
                step=-1
            
        if step is not None:
            self.movePlane(step)
            self.__drawSet()
            
            
        
        
        
    


def discreteColors(values,legendStem='n='):
    """
    Generate colors for discrete (categorical) data.

    Parameters
    ----------
    values : array of numeric
        Non-unique values to be color-coded.
    legendStem : string, optional
        Text preceding category values in legend. The default is 'n='.

    Returns
    -------
    valColors : 2d array
        Colors assigned to each input value.
    legEntries : list of matplotlib Patch
        Dummy objects to generate color legend 
        [with legend(handles=legEntry)].

    """
    colVals,colMap=np.unique(values,return_inverse=True)
    ncols=colVals.shape[0]
    
    if ncols<11:
        colSet=mpl.cm.get_cmap('tab10').colors
    else:
        colSet=mpl.cm.get_cmap('tab20').colors
    
    valColors=np.array(colSet)[colMap]
    
    legEntries=[mpl.patches.Patch(color=colSet[ii],
                               label=legendStem+'%d'%colVals[ii])
              for ii in range(colVals.shape[0])]
    
    return valColors, legEntries
              
    
def discreteLegend(axis,data,legendStem='n=',**kwargs):
    """
    Color-code data and generate corresponding legend.

    Parameters
    ----------
    axis : matplotlib Axes
        Axes to place legend in.
    data : numeric
        Data to be color-coded.
    legendStem : string, optional
        Legend text preceding values. The default is 'n='.

    Returns
    -------
    colors : array of colors
        Color corresponding to value of each data point.

    """
    colors,handles=discreteColors(data,legendStem)
    axis.legend(handles=handles,**kwargs)
    
    return colors

class SortaLogNorm(mpl.colors.SymLogNorm):
    def __init__(self, linthresh, linscale=0.1, vmin=None, vmax=None, clip=False):
        super().__init__(linthresh, linscale=linscale, vmin=None, vmax=None, clip=False)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        aval=np.abs(value)
        logval=np.sign(value)*np.log10(aval)
        
        x= [-np.log10(self.linthresh), 0, np.log10(self.linthresh), np.log10(self.vmax)]
        y=[0, 0.5*self.linscale, self.linscale, 1]
        

        return np.ma.masked_array(np.interp(logval, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y= [-np.log10(self.linthresh), 0, np.log10(self.linthresh), np.log10(self.vmax)]
        x=[0, 0.5*self.linscale, self.linscale, 1]
        
        # sign=np.ones_like(value)-np.array(value<(0.5*self.linscale))
        
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

def formatXYAxis(axis, bounds=None, symlog=False, lindist=None, xlabel='X [m]', ylabel='Y [m]'):
    """
    Set up axis for planar slices of 3d space.

    Parameters
    ----------
    axis : matplotlib Axes
        Axes to format.
    bounds : TYPE, optional
        Explicit limits for axes, [xmin, xmax, ymin, ymax]. The default is None.
    symlog : bool, optional
        Use symmetric log scales, or linear if None. The default is None.
    lindist : float, optional
        Forces linear scale on (-lindist, lindist)
    xlabel : string, optional
        Horizontal axis label. The default is 'X [m]'.
    ylabel : string, optional
        Vertical axis label. The default is 'Y [m]'.

    Returns
    -------
    None.

    """
    engform = mpl.ticker.EngFormatter()
    axis.xaxis.set_major_formatter(engform)
    axis.yaxis.set_major_formatter(engform)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if bounds is not None:
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        
    if symlog:
        axis.set_xscale('symlog', linthresh=lindist)
        axis.set_yscale('symlog',linthresh=lindist)
        
def undrawAxis(axis):
    """
    Remove contents from axis, preserving scale/labels/ticks.

    Parameters
    ----------
    axis : matplotlib Axes
        Axes to clear.

    Returns
    -------
    None.

    """
    #keep limits on redraw
    xlim=axis.get_xlim()
    ylim=axis.get_ylim()
    for col in axis.collections+axis.lines+axis.images:
        col.remove()
        
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)


def plotBoundEffect(fname, ycat='FVU', xcat='Domain size', groupby='Number of elements', legstem='%d elements'):
    dframe, _ = importRunset(fname)
    dX = dframe[xcat]
    dEl = dframe[groupby]
    yval = dframe[ycat]
    l0 = np.array([x/(n**(1/3)) for x, n in zip(dX, dEl)])

    sizes = np.unique(dEl)
    nSizes = len(sizes)
    labs = [legstem % n for n in sizes]

    def makePlot(xvals, xlabel):
        plt.figure()
        ax = plt.gca()
        ax.set_title('Simulation accuracy, uniform mesh')
        ax.grid(True)
        ax.set_ylabel('FVU')
        ax.set_xlabel(xlabel)
        outsideLegend()
        for ii in range(nSizes):
            sel = dEl == sizes[ii]
            plt.loglog(xvals[sel], yval[sel], marker='.', label=labs[ii])

        outsideLegend()
        plt.tight_layout()

    makePlot(dX, 'Domain size [m]')
    makePlot(l0, 'Element $l_0$ [m]')


def groupedScatter(fname, xcat, ycat, groupcat, filtercat=None, df=None):
    if df is None:
        df, cats = importRunset(fname)
    else:
        cats = df.keys()

    dX = df[xcat]
    dY = df[ycat]
    dL = df[groupcat]

    catNames = dL.unique()

    ax = plt.figure().add_subplot()
    ax.grid(True)
    ax.set_xlabel(xcat)
    ax.set_ylabel(ycat)
    # outsideLegend()
    for ii, label in enumerate(catNames):
        sel = dL == label
        plt.loglog(dX[sel], dY[sel], marker='.',
                   label=groupcat+': '+str(label))

    outsideLegend()
    plt.tight_layout()


def importRunset(fname):
    df = pandas.read_csv(fname)
    cats = df.keys()

    return df, cats


def importAndPlotTimes(fname, onlyCat=None, onlyVal=None, xCat='Number of elements'):
    df, cats = importRunset(fname)
    if onlyVal is not None:
        df = df[df[onlyCat] == onlyVal]

    xvals = df[xCat].to_numpy()
    nSkip = np.argwhere(cats == 'Total time')[0][0]

    def getTimes(df, cats):
        cols = cats[6:nSkip]
        return df[cols].to_numpy().transpose()

    stepTimes = getTimes(df, cats)
    stepNames = cats[6:nSkip]

    ax = plt.figure().add_subplot()

    stackedTimePlot(ax, xvals, stepTimes, stepNames)
    ax.set_xlabel(xCat)
    ax.set_ylabel("Execution time [s]")
    ax.figure.tight_layout()


def stackedTimePlot(axis, xvals, stepTimes, stepNames):
    axis.stackplot(xvals, stepTimes, baseline='zero', labels=stepNames)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # axis.figure.tight_layout()
    axis.xaxis.set_major_formatter(mpl.ticker.EngFormatter())


def outsideLegend(axis=None,**kwargs):
    """
    Create legend outside of axes, at top right.

    Parameters
    ----------
    axis : matplotlib Axes, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        Parameters passed to legend().

    Returns
    -------
    None.

    """
    if axis is None:
        axis=plt.gca()
    axis.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., **kwargs)


def showEdges(axis, coords, edgeIndices, edgeVals=None, colorbar=True,**kwargs):
    edgePts = [[coords[a, :], coords[b, :]] for a, b in edgeIndices]

    if 'colors' in kwargs:
        pass
    else:
        if edgeVals is not None:
            (cMap, cNorm) = getCmap(edgeVals)
            gColor = cMap(cNorm(edgeVals))
            kwargs['alpha'] = 1.
            kwargs['colors'] = gColor
            if colorbar:
                axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))
    
        else:
        
            kwargs['colors'] = (0., 0., 0.)
            kwargs['alpha'] = 0.05

    gCollection = p3d.art3d.Line3DCollection(
        edgePts, **kwargs)

    axis.add_collection(gCollection)
    return gCollection


def showNodes3d(axis, coords, nodeVals, cMap=None, cNorm=None,colors=None):
    """
    Show mesh nodes in 3d.
    
    Parameters
    ----------
    axis : maplotlib Axes
        Axes to plot on.
    coords : array of cartesian coordinates
        DESCRIPTION.
    nodeVals : numeric array
        Values to set node coloring.
    cMap : colormap, optional
        DESCRIPTION. The default is None.
    cNorm : matplotlib Norm, optional
        DESCRIPTION. The default is None.
    colors : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    scatterArt : TYPE
        DESCRIPTION.

    """
    if colors is None:
        if cMap is None and cNorm is None:
            (cMap, cNorm) = getCmap(nodeVals)
        vColors = cMap(cNorm(nodeVals))
        axis.figure.colorbar(mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))

    else:
        vColors=discreteLegend(axis, nodeVals)
        
    x, y, z = np.hsplit(coords, 3)

    scatterArt=axis.scatter3D(x, y, z, c=vColors)
    return scatterArt
    


def getCmap(vals, forceBipolar=False, logscale=False):
    """
    Get appropriate colormap for given continuous data.
    
    By default, uses plasma if data is mostly all positive
    or negative; otherwise, seismic is used and scaled around
    zero.
    
    Linear scaling will be used unless logscale=True, in
    which case log/symlog is automatically 

    Parameters
    ----------
    vals : numeric
        Data to be colored.
    forceBipolar : bool, optional
        DESCRIPTION. The default is False.
    logscale : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cMap : colormap
        Selected colormap.
    cNorm : matplotlib Norm
        Norm function to map data to (0,1) range of colormap.

    """
    mn = min(vals)
    mx = max(vals)
    va = abs(vals)
    knee = min(va[va > 0])
    ratio=abs(mn/mx)
    
    crosses=(mx/mn)<0

    if (crosses and (ratio > 0.01)) or forceBipolar:
        #significant data on either side of zero; use symmetric
        amax = max(abs(mn), mx)
        cMap = mpl.cm.seismic.copy()
        if logscale:
            cNorm = mpl.colors.SymLogNorm(linthresh=knee,
                                          vmin=-amax,
                                          vmax=amax)
        else:
            cNorm = mpl.colors.CenteredNorm(halfrange=amax)
    else:
        cMap = mpl.cm.plasma.copy()
        if logscale:
            cNorm = mpl.colors.LogNorm(vmin=knee,
                                        vmax=mx)
        else:
            # if mx>0:
            #     cNorm = mpl.colors.Normalize(vmin=0, vmax=mx)
            # else:
            #     cNorm = mpl.colors.Normalize(vmin=mn, vmax=0)
            cNorm=mpl.colors.Normalize(vmin=mn,vmax=mx)
    return (cMap, cNorm)


def new3dPlot(boundingBox,*args,fig=None):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    if fig is None:
        fig = plt.figure()
    axis = fig.add_subplot(*args,projection='3d')
    origin = boundingBox[:3]
    span = boundingBox[3:]-origin
    center = 0.5*(origin+boundingBox[3:])

    max_range = max(span)

    Xb = 0.5*max_range*np.mgrid[-1:2:2, -
                                1:2:2, -1:2:2][0].flatten() + 0.5*center[0]
    Yb = 0.5*max_range*np.mgrid[-1:2:2, -
                                1:2:2, -1:2:2][1].flatten() + 0.5*center[1]
    Zb = 0.5*max_range*np.mgrid[-1:2:2, -
                                1:2:2, -1:2:2][2].flatten() + 0.5*center[2]
    for xb, yb, zb in zip(Xb, Yb, Zb):
        axis.plot([xb], [yb], [zb], color=[1, 1, 1, 0])

    return axis


def animatedTitle(figure,text):
    title=figure.text(0.5,.95,text,
                   horizontalalignment='center',
                   verticalalignment='bottom')
    return title

def equalizeScale(ax):
    """
    Force 3d axes to same scale.
    
    Method based on https://stackoverflow.com/a/13701747
    
    Parameters
    ----------
    axis: matplotlib Axes
    
    Returns
    -------
    None.
    
    """
    xy = ax.xy_dataLim
    zz = ax.zz_dataLim

    max_range = np.array([xy.xmax-xy.xmin, xy.ymax-xy.ymin,
                         zz.max.max()-zz.min.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][0].flatten() + 0.5*(xy.xmax+xy.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][1].flatten() + 0.5*(xy.ymax+xy.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][2].flatten() + 0.5*(zz.max.max()+zz.min.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], color=[1, 1, 1, 0])


def showRawSlice(valList, ndiv):
    nX = ndiv+1
    nslice = nX//2

    vMat = valList.reshape((nX, nX, nX))

    vSelection = vMat[:, :, nslice].squeeze()

    (cMap, cNorm) = getCmap(vSelection.ravel())

    plt.imshow(vSelection, cmap=cMap, norm=cNorm)
    plt.colorbar()

#TODO: deprecate?
def showCurrentVecs(axis, pts, vecs):
    X, Y, Z = np.hsplit(pts, 3)
    dx, dy, dz = np.hsplit(vecs, 3)

    iMag = np.linalg.norm(vecs, axis=1)

    cMap, cNorm = getCmap(iMag)

    colors = cMap(iMag)

    art=axis.quiver3D(X, Y, Z, dx, dy, dz, colors=colors)
    return art


def showEdges2d(axis, edgePoints, edgeColors=None, **kwargs):
    """
    Show mesh edges in a 2d plot.

    Parameters
    ----------
    axis : matplotlib axis
        2d axis to plot in.
    edgePoints : List of pairs of xy coords
        .
    edgeColors : float[:], optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        Args passed to matplotlib LineCollection.

    Returns
    -------
    edgeCol : matplotlib LineCollection
        Artist for displaying the edges.

    """
    if edgeColors is None:
        kwargs['colors'] = (0., 0., 0.,)

        kwargs['alpha'] = 0.05
    else:
        kwargs['colors']=edgeColors
        # alpha=0.05

    edgeCol = mpl.collections.LineCollection(edgePoints,
                                             linewidths=0.5,
                                             **kwargs)
    axis.add_collection(edgeCol)
    return edgeCol


def showSourceBoundary(axes, radius, srcCenter=np.zeros(2)):
    """
    Plot faint ring representing source's boundary

    Parameters
    ----------
    axes : [axis]
        Axes to plot to
    radius : float
        Source radius.
    srcCenter : float[:], optional
        Center of source. The default is np.zeros(2).

    Returns
    -------
    None.

    """
    tht = np.linspace(0, 2*np.pi)
    x = radius*np.cos(tht)+srcCenter[0]
    y = radius*np.sin(tht)+srcCenter[1]

    for ax in axes:
        ax.plot(x, y, 'k', alpha=0.1)


def addInset(baseAxis, rInset, xmax, relativeLoc=(.5, -.8)):
    """
    Create sub-axis for plotting a zoomed-in view of the main axis.
    Plot commands executed in the main axis DO NOT automatically
    appear in inset; to keep them synchronized, use a pattern like

    axes=[mainAx,inset]
    for ax in axes:
        ax.plot_command(...)

    Parameters
    ----------
    baseAxis : axis
        Parent axis.
    rInset : float
        Size of inset bounds (such that bounding box is
                              rInset*[-1,1,-1,1])
    xmax : float
        Size of parent axis' bounds.
    relativeLoc : TYPE, optional
        Placement of inset's center, relative to parent axis.
        The default is (.5, -.8), which plots
        directly beneath the main plot.

    Returns
    -------
    inset : axis
        Inset axis.

    """
    insetZoom = 0.8*xmax/rInset
    inset = zoomed_inset_axes(baseAxis, zoom=insetZoom,
                              bbox_to_anchor=relativeLoc, loc='center',
                              bbox_transform=baseAxis.transAxes)

    inset.set_xlim(-rInset, rInset)
    inset.set_ylim(-rInset, rInset)
    bnds = rInset*np.array([-1, 1, -1, 1])
    formatXYAxis(inset, bnds, xlabel=None, ylabel=None)

    mark_inset(baseAxis, inset, loc1=1, loc2=2,
               fc='none', ec="0.5")

    return inset


def patchworkImage(axis, maskedArrays, cMap, cNorm, extent):
    """
    Produce composite image from discrete rectangular regions.
    
    Shows the actual interpolation structure of adaptive (non-conforming)
    grid

    Parameters
    ----------
    axis : axis
        Axis to plot on.
    quadBbox : float[:,:]
        List of [xmin, xmax, ymin, ymax] for each discrete
        rectangle.
    quadVal : float[:,:]
        DESCRIPTION.
    cMap : colormap
        Desired colormap for image.
    cNorm : norm
        Desired colormap.

    Returns
    -------
    imlist : list of artists
        Artists required to tile region.

    """
    imlist = []

    cmap = cMap.with_extremes(bad=(1., 1., 1., 0.))

    for vmask in maskedArrays:

        im = axis.imshow(vmask,
                         origin='lower', extent=extent,
                         cmap=cmap, norm=cNorm, interpolation='bilinear')
        imlist.append(im)

    return imlist

def showMesh(setup):
    ax=new3dPlot(setup.mesh.bbox)
    
    showEdges(ax, setup.mesh.nodeCoords, setup.edges,
              setup.conductances)
    x,y,z=np.hsplit(setup.mesh.nodeCoords,3)
    ax.scatter3D(x,y,z,color='k',alpha=0.5,marker='.')
        

def getPlanarEdgePoints(coords, edges, normalAxis=2, axisCoord=0):

    sel = np.equal(coords[:, normalAxis], axisCoord)
    selAx = np.array([n for n in range(3) if n != normalAxis])

    planeCoords = coords[:, selAx]

    edgePts = [[planeCoords[a, :], planeCoords[b, :]]
               for a, b in edges if sel[a] & sel[b]]

    return edgePts


def toEdgePoints(planeCoords, edges):
    edgePts = np.array([planeCoords[e] for e in edges])

    return edgePts


class FigureAnimator:
    def __init__(self, fig, study, prefs=None):
        self.fig = fig
        self.study = study
        self.axes = []

        if prefs is None:
            self.prefs = dict()
        else:
            self.prefs = prefs

        self.setupFigure()

    def setupFigure(self):
        pass

    def addSimulationData(self, sim):
        pass

    def animateStudy(self, fname=None,artists=None):
        
        if artists is None:
            artists = self.getArtists()

        animation = ArtistAnimation(self.fig,
                                    artists,
                                    interval=1000,
                                    repeat_delay=2000,
                                    blit=False)

        if fname is None:
            plt.show()
        else:
            # writer=mpl.animation.FFMpegFileWriter(fps=30)
            animation.save(os.path.join(
                self.study.studyPath, fname+'.mp4'),
                # writer=writer)
                fps=1.0)

        return animation

    def getStudyData(self, **kwargs):
        fnames = self.study.getSavedSims(**kwargs)
        for name in fnames:
            self.addSimulationData(self.study.loadData(name))

    def getArtists(self):
        pass

    def resetFigure(self):
        pass


class SliceSet(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        if prefs is None:
            prefs={
                'relativeError':False,
                'logScale':False,
                   }

        super().__init__(fig, study, prefs)

    def setupFigure(self, resetBounds=True):
        self.bnds = self.study.bbox[[0, 3, 2, 4]]
        self.grid = AxesGrid(self.fig, 211,  # similar to subplot(144)
                             nrows_ncols=(1, 2),
                             axes_pad=(0.5, .15),
                             # axes_pad=0.2,
                             label_mode="L",
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="5%",
                             cbar_pad="2%",
                             )

        self.axes = self.grid[:]
        self.axes.extend(self.grid.cbar_axes[:])

        self.grid[0].set_title('Simulated potential [V]')
        self.grid[1].set_title('Absolute error [V]')

        for ax in self.grid[:2]:
            formatXYAxis(ax, self.bnds)

        if resetBounds:
            self.vbounds = ScaleRange()
            self.errbounds = ScaleRange()
            self.rElec = 0

        self.meshEdges = []
        self.sourceEdges = []

        self.maskedVArr = []
        self.maskedErrArr = []

    def addSimulationData(self, sim):
        isNodeInPlane = sim.mesh.nodeCoords[:, -1] == 0
        intcoord = sim.intifyCoords()
        pcoord = intcoord[:, :-1]
        xyCoord = sim.mesh.nodeCoords[:, :-1]

        if len(sim.currentSources)>0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec

        if len(sim.voltageSources)>0:
            rElec=sim.voltageSources[0].radius
            self.rElec = rElec

        _,err,_,_ = sim.calculateErrors()
        err1d = err[isNodeInPlane]

        vArrays = util.coords2MaskedArrays(pcoord, sim.edges,
                                      isNodeInPlane,
                                      sim.nodeVoltages)
        erArrays = util.coords2MaskedArrays(pcoord, sim.edges,
                                       isNodeInPlane,
                                       err)

        self.maskedVArr.append(vArrays)
        self.maskedErrArr.append(erArrays)

        self.vbounds.update(sim.nodeVoltages)
        self.errbounds.update(err1d)

        filteredRoles = sim.nodeRoleTable.copy()
        filteredRoles[~isNodeInPlane] = -1
        allEdges = np.array(sim.edges)
        #TODO: deprecated function?
        roles = util.edgeRoles(allEdges, filteredRoles)

        isEdgeInPlane = ~np.any(roles == -1, axis=1)
        touchesSource = roles == 2

        isInvalid = np.all(touchesSource, axis=1) | (~isEdgeInPlane)
        isBoundaryNode = np.logical_and(touchesSource, np.sum(
            touchesSource, axis=1, keepdims=True) == 1)

        boundaryNodes = allEdges[isBoundaryNode &
                                 np.expand_dims(isEdgeInPlane, axis=1)]

        edgePoints = toEdgePoints(xyCoord, allEdges[~isInvalid])
        sourceEdge = [[np.zeros(2), xy] for xy in xyCoord[boundaryNodes]]

        self.meshEdges.append(edgePoints)
        self.sourceEdges.append(sourceEdge)

    def getArtists(self):
        vmap, vnorm = getCmap(self.vbounds.get())
        emap, enorm = getCmap(self.errbounds.get(),
                              forceBipolar=True)
        artistSet = []

        vmappr = mpl.cm.ScalarMappable(norm=vnorm, cmap=vmap)
        # vmappr.set_clim((vnorm.vmin, vnorm.vmax))

        emappr = mpl.cm.ScalarMappable(norm=enorm, cmap=emap)
        # vmappr.set_clim((-enorm.halfrange, enorm.halfrange))

        self.grid.cbar_axes[0].colorbar(
            vmappr)
        self.grid.cbar_axes[1].colorbar(emappr)

        insets = []
        if self.rElec > 0:
            for ax in self.grid:
                inset = addInset(ax, 3*self.rElec, self.bnds[1])
                showSourceBoundary([ax, inset], self.rElec)
                insets.append(inset)

        for ii in range(len(self.maskedVArr)):
            artists = []
            # vInterp=self.vArrays[ii]
            # err=self.errArrays[ii]
            edgePoints = self.meshEdges[ii]
            sourceEdge = self.sourceEdges[ii]
            # vArt=self.grid[0].imshow(vInterp, origin='lower', extent=self.bnds,
            #             cmap=vmap, norm=vnorm,interpolation='bilinear')
            # self.grid.cbar_axes[0].colorbar(vArt)
            vArt = patchworkImage(self.grid[0],
                                  self.maskedVArr[ii],
                                  vmap, vnorm,
                                  extent=self.bnds)
            errArt = patchworkImage(self.grid[1],
                                    self.maskedErrArr[ii],
                                    emap, enorm,
                                    extent=self.bnds)
            artists.extend(vArt)
            artists.extend(errArt)

            # errArt=self.grid[1].imshow(err, origin='lower', extent=self.bnds,
            #             cmap=emap, norm=enorm,interpolation='bilinear')
            # self.grid.cbar_axes[1].colorbar(errArt)

            if len(insets) > 0:
                # artists.append(insets[0].imshow(vInterp, origin='lower', extent=self.bnds,
                #             cmap=vmap, norm=vnorm,interpolation='bilinear'))
                # artists.append(insets[1].imshow(err, origin='lower', extent=self.bnds,
                #             cmap=emap, norm=enorm,interpolation='bilinear'))
                artists.extend(patchworkImage(insets[0],
                                              self.maskedVArr[ii],
                                              vmap, vnorm,
                                              extent=self.bnds))
                artists.extend(patchworkImage(insets[1],
                                              self.maskedErrArr[ii],
                                              emap, enorm,
                                              extent=self.bnds))

            for ax in self.grid:
                artists.append(showEdges2d(ax, edgePoints))

            for ax in insets:
                artists.append(showEdges2d(ax, edgePoints))
                artists.append(showEdges2d(
                    ax, sourceEdge, edgeColors=(.5, .5, .5), alpha=.25, linestyles=':'))

            self.axes.extend(insets)

            # artists.append(errArt)
            # artists.append(vArt)
            artistSet.append(artists)

        return artistSet

    def resetFigure(self):
        for ax in self.axes:
            ax.cla()


class ErrorGraph(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        if prefs is None:
            prefs={
                'showRelativeError':False,
                'colorNodeConnectivity':False,
                'onlyDoF':False}
        super().__init__(fig, study, prefs)

    def setupFigure(self):
        # axV = self.fig.add_subplot(5, 1, (1, 2))
        # axErr = self.fig.add_subplot(5, 1, (3, 4))
        # axL=self.fig.add_subplot(5,1,5)
        axV = self.fig.add_subplot(3, 1, 1)
        axErr = self.fig.add_subplot(3, 1,2)
        axL=self.fig.add_subplot(3,1,3)
        axV.grid(True)
        axErr.grid(True)
        axL.grid(True)
        
        axV.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
        axV.set_xlabel('Distance from source [m]')
        axV.set_ylabel('Voltage [V]')
        if self.prefs['showRelativeError']:
            axErr.set_ylabel('Relative error')
        else:
            axErr.set_ylabel('Absolute error [V]')
        axL.set_ylabel(r'Element $l_0$')

        axV.set_xscale('log')
        axErr.set_xscale('log')
        axL.set_xscale('log')
        axL.set_yscale('log')
        
        axErr.sharex(axV)
        axL.sharex(axV)

        self.axV = axV
        self.axErr = axErr
        self.axL=axL

        self.titles = []
        self.errR = []
        self.errors = []
        self.rColors = []
        self.simR = []
        self.sims = []
        
        self.elemR=[]
        self.elemL=[]

        self.analytic = []
        self.analyticR = []

    def addSimulationData(self, sim):
        isDoF = sim.nodeRoleTable == 0
        v = sim.nodeVoltages

        nNodes = len(sim.mesh.nodeCoords)
        nElems = len(sim.mesh.elements)
        r = np.linalg.norm(sim.mesh.nodeCoords, axis=1)

        if len(sim.currentSources)>0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec
          
        if len(sim.voltageSources)>0:
            rElec=sim.voltageSources[0].radius
            self.rElec = rElec
          
        # check how connected nodes are

        connGlobal = sim.getNodeConnectivity(True)


        lmin = np.log10(rElec/2)
        lmax = np.log10(max(r))

        if len(self.analyticR) == 0:
            rDense = np.logspace(lmin, lmax, 200)

            # _,_,vAna,_=sim.calculateErrors(rDense)
            vD,_=sim.analyticalEstimate(rDense)
            vAna=vD[0]
            self.analyticR = rDense
            self.analytic = vAna

            self.axV.set_xlim(left=rElec/2, right=2*max(r))

        ErrSum, err, vAna, sorter = sim.calculateErrors()
        
        rsort=r[sorter]
        rsort[0] = 10**lmin #for nicer plotting
        vsort=v[sorter]
        connsort = connGlobal[sorter]

        # filter non DoF
        if self.prefs['onlyDoF']:
            filt = isDoF[sorter]
        else:
            filt=np.ones_like(sorter,dtype=bool)

        rFilt = rsort[filt]
        if self.prefs['showRelativeError']:
            errRel=err/vAna
            errSort=errRel[sorter]
        else:
            errSort=err[sorter]
            
        errFilt = errSort[filt]
        connFilt = connsort[filt]

        self.titles.append('%s:%d nodes, %d elements, error %.2g' % (
            sim.meshtype, nNodes, nElems, ErrSum))
        self.errR.append(rFilt)
        self.errors.append(errFilt)
        self.rColors.append(connFilt)
        self.simR.append(rsort)
        self.sims.append(vsort)
        
        l0=np.array([e.l0 for e in sim.mesh.elements])
        center=np.array([e.origin+0.5*e.span for e in sim.mesh.elements])
        elR=np.linalg.norm(center,axis=1)
        
        self.elemL.append(l0)
        self.elemR.append(elR)

    def getArtists(self):
        artistSet = []

        self.axV.plot(self.analyticR, self.analytic, c='b', label='Analytical')

        if self.prefs['colorNodeConnectivity']:
            allconn = []
            for d in self.rColors:
                allconn.extend(d)
            connNs = np.unique(allconn)
            toN=np.zeros(max(connNs)+1,dtype=int)
            toN[connNs]=np.arange(connNs.shape[0])
            
            colors=discreteLegend(self.axErr, connNs,loc='upper right')
            # colors,legEntries=discreteColors(connNs)
            # outsideLegend(axis=self.axErr,handles=legEntries)

        for ii in range(len(self.titles)):
            artists = []
            simLine = self.axV.plot(self.simR[ii], self.sims[ii],
                                    c='k', marker='.', label='Simulated')
            if ii == 0:
                self.axV.legend(loc='upper right')
                # outsideLegend(axis=self.axV)
            # title=fig.suptitle('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
            title = self.fig.text(0.5, 1.01,
                                  self.titles[ii],
                                  horizontalalignment='center',
                                  verticalalignment='bottom',
                                  transform=self.axV.transAxes)

            if self.prefs['colorNodeConnectivity']:
                nconn = toN[self.rColors[ii]]
                nodeColors=colors[nconn]
            else:
                nodeColors='r'
            
            
            errLine = self.axErr.scatter(self.errR[ii], self.errors[ii],
                                         c=nodeColors, marker='.', linestyle='None')
            errArea=self.axErr.fill_between(self.errR[ii],
                                           self.errors[ii],
                                           color='r',
                                           alpha=0.75)


            #Third pane: element sizes
            # ln=[
            #     [[r-l/2, v], [r+l/2,v]] 
            #     for r,l,v in zip(self.elemR[ii],
            #                      self.elemL[ii],
            #                      np.arange(self.elemR[ii].shape[0]))]
            
            # l0line=mpl.collections.LineCollection(ln, colors=(0.,0.,0.))
            # self.axL.add_collection(l0line)
            l0line=self.axL.scatter(self.elemR[ii],
                                    self.elemL[ii],
                                    c='k', marker='.')

            plt.tight_layout()
            artists.append(title)
            artists.append(errLine)
            artists.append(simLine[0])
            artists.append(l0line)
            artists.append(errArea)

            artistSet.append(artists)
        return artistSet


class CurrentPlot(FigureAnimator):
    def __init__(self, fig, study, fullarrow=False,showInset=True,normalAxis=2,normalCoord=0.):
        super().__init__(fig, study)

        # inf lower bound needed for log colorbar
        self.crange = ScaleRange()
        self.cvals = []
        self.pts = []
        if fig is None:
            self.ax = new3dPlot(study.bbox)
            self.dim = 3
        else:
            self.dim = 2
            if showInset:
                self.ax = plt.subplot2grid((3, 3), (0, 1),
                                           colspan=2, rowspan=2,
                                           fig=fig)
            else:
                self.ax = fig.add_subplot()
                
            bnds = study.bbox[[0, 3, 1, 4]]
            formatXYAxis(self.ax, bnds)

        self.rElec = 0
        self.iSrc = []
        self.fullarrow = fullarrow
        self.showInset=showInset
        
        self.normalAxis=normalAxis
        self.normalCoord=normalCoord

    def addSimulationData(self, sim):
        i, E = sim.getEdgeCurrents()

        coords = sim.mesh.nodeCoords
        isSrc = sim.nodeRoleTable == 2
        whichSrc = sim.nodeRoleVals[isSrc]
        srcCoords = np.array([s.coords for s in sim.currentSources])
        coords[isSrc] = srcCoords[whichSrc]


        if len(sim.currentSources)>0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec
        
        if len(sim.voltageSources)>0:
            rElec=sim.voltageSources[0].radius
            self.rElec = rElec

        self.iSrc.append(sim.currentSources[0].value)

        if self.dim == 3:
            eplane = E
            iplane = i

        else:
            #only get points in specified plane
            centerCoord=np.zeros(3)
            centerCoord[self.normalAxis]=self.normalCoord
            level=sim.intifyCoords(centerCoord)[self.normalAxis]
            
            ptInPlane = sim.intifyCoords()[:, self.normalAxis] == level
            edgeInPlane = [np.all(ptInPlane[e]) for e in E]

            eplane = E[edgeInPlane]
            iplane = i[edgeInPlane]

        # discard zero currents
        isok = iplane > 0
        iOK = iplane[isok]
        ptOK = coords[eplane[isok]]

        self.cvals.append(iOK)
        self.pts.append(ptOK)

        self.crange.update(iOK)

    def getArtists(self):
        # include injected currents in scaling
        self.crange.update(np.array(self.iSrc))
        cmap, cnorm = getCmap(self.crange.get(), logscale=True)
        # cmap=mpl.cm.YlOrBr
        # cnorm=mpl.colors.LogNorm(vmin=self.crange[0],
        #                           vmax=self.crange[1])
        plt.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                     ax=self.ax)
        plt.title('Edge currents')
        inset = None
        if self.showInset & (self.rElec > 0) & (self.dim != 3):
            inset = addInset(self.ax,
                             3*self.rElec,
                             self.study.bbox[3],
                             (-.2, -.2))
            showSourceBoundary([self.ax, inset],
                               self.rElec)

        artists = []
        for ii in range(len(self.pts)):
            pt = self.pts[ii]
            cv = self.cvals[ii]
            colors = cmap(cnorm(cv))

            x0 = pt[:, 0, :].squeeze()
            d0 = pt[:, 1, :].squeeze()-x0
            m0 = x0+0.5*d0
            mags = d0/np.linalg.norm(d0, axis=1, keepdims=True)

            x, y, z = np.hsplit(x0, 3)
            a, b, c = np.hsplit(d0, 3)
            m1, m2, m3 = np.hsplit(m0, 3)
            v1, v2, v3 = np.hsplit(mags, 3)
            if self.dim == 3:
                artists.append([
                    self.ax.quiver3D(x, y, z, a, b, c,
                                     colors=colors)
                ])

            else:
                axes = [self.ax]
                art = []
                if inset is not None:
                    axes.append(inset)

                for ax in axes:
                    if self.fullarrow:
                        art.append(ax.quiver(x, y, a, b,
                                             color=colors,
                                             angles='xy',
                                             scale_units='xy',
                                             scale=1))#,
                                             # headlength=15,
                                             # headwidth=10))
                    else:
                        #TODO: hack to get edges; assumes xy plane at z=0
                        art.append(showEdges2d(ax, 
                                               pt[:,:,:-1],
                                               edgeColors=(0.,0.,0.),
                                               alpha=0.25))
                        art.append(ax.quiver(m1, m2,
                                             a/2, b/2,
                                             color=colors,
                                             pivot='mid',
                                             angles='xy',
                                             scale_units='xy',
                                             scale=1))
                        # headlength=15,
                        # headwidth=10))

                    art.append(ax.scatter(0, 0,
                                          color=cmap(cnorm(self.iSrc[ii]))))

                artists.append(art)

        return artists


class ScaleRange:
    def __init__(self, vals=None):
        self.min = np.inf
        self.max = np.NINF
        self.knee = np.inf
        if vals is not None:
            self.update(vals)

    def update(self, newVals):
        if newVals is not None:
            va = abs(newVals)
            self.min = min(self.min, min(newVals))
            self.max = max(self.max, max(newVals))
            self.knee = min(self.knee, min(va[va > 0]))

    def get(self):
        isBipolar = (self.min < 0) & (self.max > 0)
        if isBipolar:
            out = np.array([self.min, self.knee, self.max])
        else:
            out = np.array([self.min, self.max])
        return out





def showRegularization(sim):
    fig=plt.figure()
    # axAdd,axSub=fig.add_subplots
    axAdd=new3dPlot(sim.mesh.bbox,1,2,1,fig=fig)
    axSub=new3dPlot(sim.mesh.bbox,1,2,2,fig=fig)
    
    oldMat=sim.getEdgeMat()
    sim.regularizeMesh()
    newMat=sim.getEdgeMat()
    
    dif=tril(newMat-oldMat)
    dEdge=np.array(dif.nonzero()).transpose()
    dG=dif.data
    cmap,cnorm=getCmap(dG,forceBipolar=True)

    
    isNew=dG>0
    
    addG=dG[isNew]
    addE=dEdge[isNew]
    subG=dG[~isNew]
    subE=dEdge[~isNew]
    
    coords=sim.mesh.nodeCoords
    
    addPts=coords[addE.ravel()]
    subPts=sim.mesh.nodeCoords[subE.ravel()]
    addCol=np.repeat(addG,2)
    subCol=np.repeat(subG,2)
    
    ax,ay,az=np.hsplit(addPts,3)
    sx,sy,sz=np.hsplit(subPts,3)
    
    showEdges(axAdd,coords, addE,addG,colorbar=False)
    showEdges(axSub,coords,subE,subG,colorbar=False)
    
    # fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=cnorm))
    
    axAdd.scatter3D(ax,ay,az,c=addCol)
    axSub.scatter3D(sx,sy,sz,c=subCol)
    
    axAdd.set_title('Edges added')
    axSub.set_title('Edges removed')