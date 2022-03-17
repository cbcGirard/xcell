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
from matplotlib.ticker import EngFormatter as eform
from matplotlib.animation import ArtistAnimation, FFMpegWriter
import numpy as np
import numba as nb
# from scipy.interpolate import interp2d
from scipy.sparse import tril
import os

# import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

import pickle
import pandas
# from util import uniformResample, edgeRoles, getquads, quadsToMaskedArrays, coords2MaskedArrays
import util


# Dark mode
FAINT = (1., 1., 1., 0.05)
CM_BIPOLAR = mpl.colors.LinearSegmentedColormap.from_list('bipolar',
                                                          np.array([[0, 0, 1, 1],
                                                                 [.098, .137, .176, 0],
                                                                 [1, 0, 0, 1]],
                                                                dtype=float))
BASE = '#afcfff'


class TimingBar:
    def __init__(self, figure, axis, data=None):
        self.maxtime = np.inf
        self.fig = figure
        if axis is None:
            self.ax = figure.add_subplot()
        else:
            self.ax = axis

        # self.ax.set_xlabel('Time [ms]')
        self.ax.xaxis.set_major_formatter(eform('s'))

        [self.ax.spines[k].set_visible(False) for k in self.ax.spines]

        self.data = data
        if data is not None:
            if data['style'] != 'sweep':
                self.ax.plot(data['x'], data['y'], color='C0')
            self.ax.set_ylabel(data['ylabel'])
            self.ax.yaxis.set_major_formatter(eform(data['unit']))
            self.ax.set_yticks([0])
            self.ax.grid(False)
            self.ax.grid(visible=True,
                         which='major',
                         axis='y')

        else:
            self.ax.yaxis.set_visible(False)

    def getArt(self, time, step=None):
        ymin, ymax = self.ax.get_ylim()
        art = []

        if step is not None:
            if self.data['style'] == 'sweep':
                art.append(self.ax.plot(self.data['x'][:step],
                                        self.data['y'][:step],
                                        color='C0'
                                        )[0])
            elif self.data['style'] == 'rave':
                art.append(self.ax.plot(self.data['x'][:step],
                                        self.data['y'][:step],
                                        )[0])
            elif self.data['style'] == 'dot':
                art.append(self.ax.scatter(self.data['x'][step],
                                           self.data['y'][step],
                                           marker='o',
                                           color='C1'))

        else:
            art.append(self.ax.vlines(time, .8*ymin, .8*ymax,
                                      colors=BASE,
                                      linewidths=.5))

        return art


class DataPrefs:
    def __init__(self, name,
                 dataSource=None,
                 scaleStyle=None,
                 cmap=None,
                 norm=None,
                 colors=None):
        self.name = name
        self.dataSource = dataSource
        self.scaleStyle = scaleStyle
        self.cmap = cmap
        self.norm = norm
        self.colors = colors
        self.range = ScaleRange()
        self.scaleViz = None
        self.dataViz = None

    def updateRange(self, newVals):
        self.range.update(newVals)

    def applyPrefs(self, axis, data):
        self.updateRange(data)
        plotargs = {}

        if self.scaleStyle == 'legend':
            if self.colors is None:
                self.colors = discreteLegend(axis, data)
            plotargs['colors'] = self.colors

        if self.scaleStyle == 'colorbar':
            if self.cmap is None:
                cmap, cnorm = getCmap(data)
            else:
                cmap = self.cmap
                cnorm = self.norm
            plotargs['cmap'] = cmap
            plotargs['norm'] = cnorm

            # plt.colorbar(mpl.cm.ScalarMappable(**plotargs),
            #              ax=axis)

        return plotargs


class DisplayPrefs:
    def __init__(self, nodePrefs=None,
                 edgePrefs=None,
                 imgPrefs=None):
        if nodePrefs is None:
            nodePrefs = DataPrefs('nodePrefs')
        if edgePrefs is None:
            edgePrefs = DataPrefs('edgePrefs')
        if imgPrefs is None:
            imgPrefs = DataPrefs('imgPrefs')

        self.nodePrefs = nodePrefs
        self.edgePrefs = edgePrefs
        self.imgPrefs = imgPrefs


class SliceViewer:
    def __init__(self, axis, sim, displayPrefs=None, topoType='mesh'):
        if axis is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot()
        else:
            self.ax = axis
            self.fig = axis.figure

        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.sim = sim
        self.nLevels = 0

        formatXYAxis(self.ax, bounds=None)
        self.topoType = topoType

        self.setPlane()
        # self.drawFns=[]
        # self.drawArgs=[]

        if displayPrefs is None:
            displayPrefs = DisplayPrefs()

        self.prefs = displayPrefs
        self.edgeData = None
        self.nodeData = None
        self.edgeArtist = None
        self.nodeArtist = None
        self.edgeScaleViz = None
        self.nodeScaleViz = None
        # self.edgeColors=[]
        # self.nodeColors=[]

    def changeData(self, newSim):
        self.sim = newSim

        self.setPlane(self.normAxis, self.planeZ)

    def setPlane(self, normalAxis=2, normCoord=0, showAll=False):

        coords = self.sim.getCoords(self.topoType)
        ic = self.sim.intifyCoords(coords)
        self.intCoord = ic

        pointCart = np.zeros(3)
        pointCart[normalAxis] = normCoord

        levelIndex = self.sim.intifyCoords(pointCart)[normalAxis]
        self.lvlIndex = levelIndex
        # self.nLevels=np.unique(ic[:,normalAxis]).shape[0]
        self.nLevels = max(ic[:, normalAxis])

        self.normAxis = normalAxis
        self.__setmask(showAll)

    def movePlane(self, stepAmount):
        self.lvlIndex += stepAmount
        self.__setmask()

    def __setmask(self, showAll=False):
        axNames = ['x', 'y', 'z']
        axlist = np.arange(3)

        coords = self.sim.getCoords(self.topoType)
        otherAxes = axlist != self.normAxis

        if showAll:
            inPlane = np.ones(self.intCoord.shape[0], dtype=np.bool_)
        else:
            inPlane = self.intCoord[:, self.normAxis] == self.lvlIndex
        zval = coords[inPlane, self.normAxis][0]
        self.planeZ = zval
        self.nInPlane = inPlane

        self.xyCoords = coords[:, otherAxes]
        self.zCoords = coords[:, ~otherAxes].squeeze()
        self.pCoords = self.xyCoords[inPlane]

        # graph formatting
        xyNames = [axNames[i]+" [m]" for i in axlist[otherAxes]]
        titleStr = axNames[self.normAxis]+"=%.2g" % zval
        self.ax.set_xlabel(xyNames[0])
        self.ax.set_ylabel(xyNames[1])
        self.ax.set_title(titleStr)

        bndInds = [d+3*ii for d in axlist[otherAxes] for ii in range(2)]
        bnds = self.sim.mesh.bbox[bndInds]
        self.ax.set_xlim(bnds[0], bnds[1])
        self.ax.set_ylim(bnds[2], bnds[3])

    def showEdges(self, connAngle=None, colors=None, **kwargs):
        # edge=self.sim.getEdges(self.topoType)
        # edgeMask=self.nInPlane[edge]
        # edgeInPlane=np.all(edgeMask,axis=1)

        # planeEdges=edge[edgeInPlane]

        # edgePts=toEdgePoints(self.xyCoords, planeEdges)

        _, _, edgePts = self.sim.getElementsInPlane(self.normAxis, self.planeZ)

        # if colors is not None:
        #     colors=colors[edgeInPlane]

        art = showEdges2d(self.ax,
                          edgePts,
                          colors,
                          **kwargs)

        # if connAngle is not None:
        #     edgeAdjacent=np.logical_xor(edgeMask[:,0],
        #                                 edgeMask[:,1])
        #     dz=np.array(self.zCoords-self.planeZ,
        #                 ndmin=2).transpose()

        #     skew=np.matmul(dz,
        #                    np.array([np.cos(connAngle),np.sin(connAngle)],
        #                             ndmin=2))
        #     skewPoints=toEdgePoints(self.xyCoords+skew,
        #                             self.sim.getEdges()[edgeAdjacent])
        #     showEdges2d(self.ax, skewPoints, colors, **kwargs)

        return art

    def showNodes(self, nodeVals, colors=None, **kwargs):

        x, y = np.hsplit(self.pCoords, 2)

        if colors is not None:
            art = self.ax.scatter(x, y,
                                  color=colors[self.nInPlane],
                                  **kwargs)
        else:
            if nodeVals is not None:
                vals = nodeVals[self.nInPlane]
            else:
                vals = 'k'
            art = self.ax.scatter(x, y, c=vals, **kwargs)

        return art

    def __drawSet(self):
        # keep limits on redraw
        undrawAxis(self.ax)

        if self.prefs.edgePrefs.name is not None:
            plotprefs = self.prefs.edgePrefs.applyPrefs(
                self.ax, self.edgeData)
            self.edgeArtist = self.showEdges(**plotprefs)

        if self.prefs.nodePrefs.name is not None:
            plotprefs = self.prefs.nodePrefs.applyPrefs(
                self.ax, self.nodeData)
            self.nodeArtist = self.showNodes(self.nodeData, **plotprefs)

        self.fig.canvas.draw()

    def onKey(self, event):
        step = None
        if event.key == 'up':
            if self.lvlIndex < self.nLevels:
                step = 1
        if event.key == 'down':
            if self.lvlIndex > 0:
                step = -1

        if step is not None:
            self.movePlane(step)
            self.__drawSet()


def discreteColors(values, legendStem='n='):
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
    colVals, colMap = np.unique(values, return_inverse=True)
    ncols = colVals.shape[0]

    if ncols < 11:
        colSet = mpl.cm.get_cmap('tab10').colors
    else:
        colSet = mpl.cm.get_cmap('tab20').colors

    valColors = np.array(colSet)[colMap]

    legEntries = [mpl.patches.Patch(color=colSet[ii],
                                    label=legendStem+'%d' % colVals[ii])
                  for ii in range(colVals.shape[0])]

    return valColors, legEntries


def discreteLegend(axis, data, legendStem='n=', **kwargs):
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
    colors, handles = discreteColors(data, legendStem)
    axis.legend(handles=handles, **kwargs)

    return colors


class SortaLogNorm(mpl.colors.SymLogNorm):
    def __init__(self, linthresh, linscale=0.1, vmin=None, vmax=None, clip=False):
        super().__init__(linthresh, linscale=linscale, vmin=None, vmax=None, clip=False)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        aval = np.abs(value)
        logval = np.sign(value)*np.log10(aval)

        x = [-np.log10(self.linthresh), 0,
             np.log10(self.linthresh), np.log10(self.vmax)]
        y = [0, 0.5*self.linscale, self.linscale, 1]

        return np.ma.masked_array(np.interp(logval, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y = [-np.log10(self.linthresh), 0,
             np.log10(self.linthresh), np.log10(self.vmax)]
        x = [0, 0.5*self.linscale, self.linscale, 1]

        # sign=np.ones_like(value)-np.array(value<(0.5*self.linscale))

        return np.interp(value, x, y, left=-np.inf, right=np.inf)


def formatXYAxis(axis, bounds=None, symlog=False, lindist=None, axlabels=False, xlabel='X [m]', ylabel='Y [m]'):
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
    engform = mpl.ticker.EngFormatter('m')
    axis.xaxis.set_major_formatter(engform)
    axis.yaxis.set_major_formatter(engform)

    if axlabels:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    axis.grid(False)
    if bounds is not None:
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])

    if symlog:
        axis.set_xscale('symlog', linthresh=lindist)
        axis.set_yscale('symlog', linthresh=lindist)


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
    # keep limits on redraw
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
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


def plotStudyPerformance(study, **kwargs):
    fig1, axes = plt.subplots(2, 1)
    fn = study.studyPath+'/log.csv'

    for ax, ttype in zip(axes, ['Wall', 'CPU']):
        importAndPlotTimes(fn, ttype, ax, **kwargs)

    fig2, _ = importAndPlotTimes(fn, timeType='Ratio', ax=None, **kwargs)

    return [fig1, fig2]


def importAndPlotTimes(fname, timeType='Wall', ax=None, onlyCat=None, onlyVal=None, xCat='Number of elements'):
    df, cats = importRunset(fname)
    if onlyVal is not None:
        df = df[df[onlyCat] == onlyVal]

    xvals = df[xCat].to_numpy()

    if timeType == 'Ratio':
        cwall = [c for c in cats if c.find('Wall') > 0]
        cCPU = [c for c in cats if c.find('CPU') > 0]
        tcpu = df[cCPU].to_numpy().transpose()
        twall = df[cwall].to_numpy().transpose()

        stepTimes = tcpu/twall
        tcols = cCPU

    else:
        tcols = [c for c in cats if c.find(
            timeType) > 0 and c.find('Total') < 0]

        stepTimes = df[tcols].to_numpy().transpose()
    stepNames = [c[:c.find('[')-1] for c in tcols]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    if timeType == 'Ratio':
        for val, lbl in zip(stepTimes, stepNames):
            ax.plot(xvals, val, label=lbl)

        ax.set_ylabel('Estimated parallel speedup')
        ax.set_xlabel(xCat)
        outsideLegend(ax)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())

    else:
        stackedTimePlot(ax, xvals, stepTimes, stepNames)
        ax.set_xlabel(xCat)
        ax.set_ylabel(timeType+" time [s]")

    ax.figure.tight_layout()

    return fig, ax


def stackedTimePlot(axis, xvals, stepTimes, stepNames):
    axis.stackplot(xvals, stepTimes, baseline='zero', labels=stepNames)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    outsideLegend(axis)
    # axis.figure.tight_layout()
    axis.xaxis.set_major_formatter(mpl.ticker.EngFormatter())


def outsideLegend(axis=None, **kwargs):
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
        axis = plt.gca()
    axis.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=0., **kwargs)


def showEdges(axis, coords, edgeIndices, edgeVals=None, colorbar=True, **kwargs):
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
                axis.figure.colorbar(
                    mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap))

        else:

            kwargs['colors'] = BASE
            kwargs['alpha'] = 0.05

    gCollection = p3d.art3d.Line3DCollection(
        edgePts, **kwargs)

    axis.add_collection(gCollection)
    return gCollection


def showNodes3d(axis, coords, nodeVals, cMap=None, cNorm=None, colors=None):
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
        vColors = discreteLegend(axis, nodeVals)

    x, y, z = np.hsplit(coords, 3)

    scatterArt = axis.scatter3D(x, y, z, c=vColors)
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
    if any(va > 0):
        knee = min(va[va > 0])
    else:
        knee = 0
    ratio = abs(mn/mx)

    crosses = (mx/mn) < 0

    if (crosses and (ratio > 0.01)) or forceBipolar:
        # significant data on either side of zero; use symmetric
        amax = max(abs(mn), mx)
        # cMap = mpl.cm.seismic.copy()
        # clist=[[0.0, 0.0, 1.0, 1.0],
        #        [0.5, 0.5, 1.0, 1.0],
        #        [1.0, 1.0, 1.0, 0.0],
        #        [1.0, 0.5, 0.5, 1.0],
        #        [1.0, 0.0, 0.0, 1.0]]
        # cMap=mpl.colors.LinearSegmentedColormap.from_list('biBR', clist)
        cMap = CM_BIPOLAR
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
            cNorm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    return (cMap, cNorm)


def new3dPlot(boundingBox, *args, fig=None):
    """ Force 3d axes to same scale
        Based on https://stackoverflow.com/a/13701747
    """
    if fig is None:
        fig = plt.figure()
    axis = fig.add_subplot(*args, projection='3d')
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


def animatedTitle(figure, text):
    title = figure.text(0.5, .95, text,
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

# TODO: deprecate?


def showCurrentVecs(axis, pts, vecs):
    X, Y, Z = np.hsplit(pts, 3)
    dx, dy, dz = np.hsplit(vecs, 3)

    iMag = np.linalg.norm(vecs, axis=1)

    cMap, cNorm = getCmap(iMag)

    colors = cMap(iMag)

    art = axis.quiver3D(X, Y, Z, dx, dy, dz, colors=colors)
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
        # kwargs['colors'] = (0., 0., 0.,)

        # kwargs['alpha'] = 0.05
        kwargs['colors'] = FAINT
    else:
        kwargs['colors'] = edgeColors
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
    ax = new3dPlot(setup.mesh.bbox)

    mcoord, medge = setup.getMeshGeometry()

    showEdges(ax, mcoord, medge)

    # x,y,z=np.hsplit(setup.mesh.nodeCoords,3)
    # ax.scatter3D(x,y,z,color='k',alpha=0.5,marker='.')
    return ax


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


# class Animator:
#     def __init__(self,filename=None):
#         self.dataFrames=[]
#         self.fig=plt.figure()
#         self.axes=[]
#         self.filename=filename

#     def makeFig(self):
#         pass

#     def getFrameData(self):
#         pass

#     def getArtists(self):
#         pass

#     def animate(self,filename=None):

#         animation = ArtistAnimation(self.fig,
#                                     artists,
#                                     interval=1000,
#                                     repeat_delay=2000,
#                                     blit=False)
#         if fname is None:
#             plt.show()
#         else:
#             # writer=mpl.animation.FFMpegFileWriter(fps=30)
#             animation.save(os.path.join(
#                 self.study.studyPath, fname+'.mp4'),
#                 # writer=writer)
#                 fps=1.0)

#         return animation


class FigureAnimator:
    def __init__(self, fig, study, prefs=None):
        self.fig = fig
        self.study = study
        self.axes = []

        if prefs is None:
            self.prefs = dict()
        else:
            self.prefs = prefs

        self.dataSets = []
        self.dataCat = []
        self.dataScales = {}

        self.setupFigure()

    def __getstate__(self):
        state = self.__dict__.copy()

        state['fig'] = None
        state['axes'] = []

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.fig = plt.figure()

    def setupFigure(self):
        pass

    def addSimulationData(self, sim, append=False):
        pass

    def animateStudy(self, fname=None, artists=None, fps=1.0):

        animations = []
        artists = []
        for ii in range(len(self.dataSets)):
            artists.append(self.getArtists(ii))

        animation = ArtistAnimation(self.fig,
                                    artists,
                                    interval=1000/fps,
                                    repeat_delay=2000,
                                    blit=True)

        if fname is None:
            plt.show()

        else:
            fstem = os.path.join(self.study.studyPath,
                                 fname)
            # writer=mpl.animation.FFMpegFileWriter(fps=30)
            animation.save(fstem+'.mp4',
                           # writer=writer)
                           fps=fps)
            pickle.dump(self.__dict__, open(fstem+'.aData', 'wb'))
        animations.append(animation)

        return animation

    def getStudyData(self, **kwargs):

        fnames, cats = self.study.getSavedSims(**kwargs)
        for names, cat in zip(fnames, cats):
            dset = []
            dcat = []
            for name in names:
                # for ii in nb.prange(len(names)):
                sim = self.study.loadData(name)
                # dset.append(self.addSimulationData(sim))
                self.addSimulationData(sim, append=True)
                dcat.append(cat)

            # self.dataSets=dset
            self.dataCat = dcat

            # self.addSimulationData(self.study.loadData(name),dataLabel=dname)

    def getArtists(self, setnumber=None, data=None):
        pass

    def resetFigure(self):
        for axis in self.axes:
            undrawAxis(axis)

    def unifyScales(self, otherScales):
        for key in otherScales.keys():
            self.dataScales[key].update(otherScales[key].get())


class SliceSet(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        if prefs is None:
            prefs = {
                'relativeError': False,
                'logScale': False,
            }

        super().__init__(fig, study, prefs)
        self.dataScales = {
            'vbounds': ScaleRange(),
            'errbounds': ScaleRange()}

    def setupFigure(self, resetBounds=False):
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
            self.dataScales['vbounds'] = ScaleRange()
            self.dataScales['errbounds'] = ScaleRange()
            self.rElec = 0

        # self.meshEdges = []
        # self.sourceEdges = []

        # self.maskedVArr = []
        # self.maskedErrArr = []

    def addSimulationData(self, sim, append=False):

        if len(sim.currentSources) > 0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec

        if len(sim.voltageSources) > 0:
            rElec = sim.voltageSources[0].radius
            self.rElec = rElec

        vArrays, coords = sim.getValuesInPlane(data=None)
        cartCoords = np.hstack((coords, np.zeros((coords.shape[0], 1))))

        # TODO: better filtering to only calculate err where needed
        ana, _ = sim.analyticalEstimate()
        err1d = sim.nodeVoltages-ana[0]

        erArrays, _ = sim.getValuesInPlane(data=err1d)

        v1d = util.unravelArraySet(vArrays)

        # self.maskedVArr.append(vArrays)
        # self.maskedErrArr.append(erArrays)

        self.dataScales['vbounds'].update(v1d)
        self.dataScales['errbounds'].update(err1d)

        _, _, edgePts = sim.getElementsInPlane()

        # TODO: dehackify so it works with sources other than axes origin
        inSource = np.linalg.norm(
            edgePts, axis=2) <= sim.currentSources[0].radius
        touchesSource = np.logical_xor(inSource[:, 0], inSource[:, 1])
        edgePoints = edgePts[~touchesSource]
        sourceEdge = edgePts[touchesSource]

        # self.meshEdges.append(edgePoints)
        # self.sourceEdges.append(sourceEdge)

        data = {
            'vArrays': vArrays,
            'errArrays': erArrays,
            'meshPoints': edgePoints,
            'sourcePoints': sourceEdge}

        if append:
            self.dataSets.append(data)

        return data

    def getArtists(self, setnum, data=None):

        # if data is None:
        #     dataSet=self.dataSets[setnum]
        # else:
        #     dataSet=[data]

        if data is None:
            data = self.dataSets[setnum]

        vbounds = self.dataScales['vbounds']
        errbounds = self.dataScales['errbounds']

        vmap, vnorm = getCmap(vbounds.get())
        emap, enorm = getCmap(errbounds.get(),
                              forceBipolar=True)

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

        # for ii in range(len(self.maskedVArr)):
        # for data in dataSet:
        artists = []

        # edgePoints = self.meshEdges[ii]
        # sourceEdge = self.sourceEdges[ii]

        vArt = patchworkImage(self.grid[0],
                              data['vArrays'],
                              vmap, vnorm,
                              extent=self.bnds)
        errArt = patchworkImage(self.grid[1],
                                # self.maskedErrArr[ii],
                                data['errArrays'],
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
                                          # self.maskedVArr[ii],
                                          data['vArrays'],
                                          vmap, vnorm,
                                          extent=self.bnds))
            artists.extend(patchworkImage(insets[1],
                                          # self.maskedErrArr[ii],
                                          data['errArrays'],
                                          emap, enorm,
                                          extent=self.bnds))

        for ax in self.grid:
            artists.append(showEdges2d(ax, data['meshPoints']))

        for ax in insets:
            artists.append(showEdges2d(ax, data['meshPoints']))
            artists.append(showEdges2d(
                ax, data['sourcePoints'], edgeColors=(.5, .5, .5), alpha=.25, linestyles=':'))

        self.axes.extend(insets)

        # artists.append(errArt)
        # artists.append(vArt)
        # artistSet.append(artists)

        return artists

    # def resetFigure(self):
    #     for ax in self.axes:
    #         ax.cla()


class ErrorGraph(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        if prefs is None:
            prefs = {
                'showRelativeError': False,
                'colorNodeConnectivity': False,
                'onlyDoF': False,
                'universalPts': False}
        super().__init__(fig, study, prefs)

        self.dataScales = {
            'vsim': ScaleRange(),
            'error': ScaleRange()}

    def setupFigure(self):
        # axV = self.fig.add_subplot(5, 1, (1, 2))
        # axErr = self.fig.add_subplot(5, 1, (3, 4))
        # axL=self.fig.add_subplot(5,1,5)
        axV = self.fig.add_subplot(3, 1, 1)
        axErr = self.fig.add_subplot(3, 1, 2)
        axL = self.fig.add_subplot(3, 1, 3)
        axV.grid(True)
        axErr.grid(True)
        axL.grid(True)

        axV.xaxis.set_major_formatter(mpl.ticker.EngFormatter('m'))
        axV.yaxis.set_major_formatter(mpl.ticker.EngFormatter('V'))
        axL.yaxis.set_major_formatter(mpl.ticker.EngFormatter('m'))


        axV.set_xlabel('Distance from source')
        axV.set_ylabel('Voltage')


        if self.prefs['showRelativeError']:
            axErr.set_ylabel('Relative error')
        else:
            axErr.set_ylabel('Absolute error')
            axErr.yaxis.set_major_formatter(mpl.ticker.EngFormatter('V'))
        axL.set_ylabel(r'Element $l_0$')

        axV.set_xscale('log')
        axErr.set_xscale('log')
        axL.set_xscale('log')
        axL.set_yscale('log')

        axErr.sharex(axV)
        axL.sharex(axV)

        self.axV = axV
        self.axErr = axErr
        self.axL = axL

        self.axes = [axV, axErr, axL]

        # self.titles = []
        # self.errR = []
        # self.errors = []
        # self.rColors = []
        # self.simR = []
        # self.sims = []

        # self.elemR=[]
        # self.elemL=[]

        self.analytic = []
        self.analyticR = []

    def addSimulationData(self, sim, append=False):
        isDoF = sim.nodeRoleTable == 0
        # v = sim.nodeVoltages

        if self.prefs['universalPts']:
            pts, v = sim.getUniversalPoints()
            coords = util.indexToCoords(pts,
                                        sim.mesh.bbox[:3],
                                        sim.mesh.span)
            # sim.nodeVoltages = v
        else:
            v = sim.nodeVoltages
            coords = sim.mesh.nodeCoords
            pts=sim.mesh.indexMap

        nNodes = len(sim.mesh.nodeCoords)
        nElems = len(sim.mesh.elements)

        r = np.linalg.norm(coords, axis=1)

        if len(sim.currentSources) > 0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec

        if len(sim.voltageSources) > 0:
            rElec = sim.voltageSources[0].radius
            self.rElec = rElec

        lmin = np.log10(rElec/2)
        lmax = np.log10(max(r))

        if len(self.analyticR) == 0:
            rDense = np.logspace(lmin, lmax, 200)

            # _,_,vAna,_=sim.calculateErrors(rDense)
            vD, _ = sim.analyticalEstimate(rDense)
            vAna = vD[0]
            self.analyticR = rDense
            self.analytic = vAna

            self.dataScales['vsim'].update(vAna)

            self.axV.set_xlim(left=rElec/2, right=2*max(r))

        ErrSum, err, vAna, sorter,_ = sim.calculateErrors(pts)

        rsort = r[sorter]
        rsort[0] = 10**lmin  # for nicer plotting
        vsort = v[sorter]

        # filter non DoF
        if self.prefs['onlyDoF']:
            filt = isDoF[sorter]
        else:
            filt = np.ones_like(sorter, dtype=bool)
            # filt=sim.nodeRoleTable>=0

        rFilt = rsort[filt]
        if self.prefs['showRelativeError']:
            errRel = err/vAna
            errSort = errRel[sorter]
        else:
            errSort = err[sorter]

        errFilt = errSort[filt]

        title = '%s:%d nodes, %d elements, error %.2g' % (
            sim.meshtype, nNodes, nElems, ErrSum)

        # self.titles.append(title)
        # self.errR.append(rFilt)
        # self.errors.append(errFilt)

        # self.simR.append(rsort)
        # self.sims.append(vsort)

        l0 = np.array([e.l0 for e in sim.mesh.elements])
        center = np.array([e.origin+0.5*e.span for e in sim.mesh.elements])
        elR = np.linalg.norm(center, axis=1)

        # self.elemL.append(l0)
        # self.elemR.append(elR)

        data = {
            'title': title,
            'errR': rFilt,
            'errors': errFilt,
            'simR': rsort,
            'simV': vsort,
            'elemL': l0,
            'elemR': elR}

        self.dataScales['vsim'].update(vsort)
        self.dataScales['error'].update(errFilt)

        if self.prefs['colorNodeConnectivity']:
            # check how connected nodes are
            connGlobal = sim.getNodeConnectivity(True)
            connsort = connGlobal[sorter]
            connFilt = connsort[filt]
            self.rColors.append(connFilt)
            data['rColors'] = connFilt

        if append:
            self.dataSets.append(data)
        return data

    def getArtists(self, setnum, data=None):
        artistSet = []

        if data is None:
            #     dataset=[self.dataSets[setnum]]
            # else:
            #     dataset=[data]
            data = self.dataSets[setnum]

        self.axV.plot(self.analyticR, self.analytic,
                      c=BASE, label='Analytical')

        vbnd = self.dataScales['vsim'].get()
        self.axV.set_ylim(vbnd[0], vbnd[-1])

        errbnd = self.dataScales['error'].get()
        self.axErr.set_ylim(errbnd[0], errbnd[-1])

        if self.prefs['colorNodeConnectivity']:
            allconn = []
            for d in self.rColors:
                allconn.extend(d)
            connNs = np.unique(allconn)
            toN = np.zeros(max(connNs)+1, dtype=int)
            toN[connNs] = np.arange(connNs.shape[0])

            colors = discreteLegend(self.axErr, connNs, loc='upper right')
            # colors,legEntries=discreteColors(connNs)
            # outsideLegend(axis=self.axErr,handles=legEntries)

        # for ii in range(len(self.titles)):
        # for data in dataset:

        artists = []
        simLine = self.axV.plot(
            # self.simR[ii], self.sims[ii],
            data['simR'], data['simV'],
            c='C1', marker='.', label='Simulated')
        if setnum == 0:
            self.axV.legend(loc='upper right')

            # outsideLegend(axis=self.axV)
        # title=fig.suptitle('%d nodes, %d elements\nFVU= %g'%(nNodes,nElems,FVU))
        # title = self.fig.text(0.5, 1.01,
        #                       self.titles[ii],
        #                       horizontalalignment='center',
        #                       verticalalignment='bottom',
        #                       transform=self.axV.transAxes)
        title = animatedTitle(self.fig,  # self.titles[ii])
                              data['title'])

        if self.prefs['colorNodeConnectivity']:
            rcol = data['rColors']
            nconn = toN[rcol]
            nodeColors = colors[nconn]
        else:
            nodeColors = 'r'

        errLine = self.axErr.scatter(  # self.errR[ii], self.errors[ii],
            data['errR'], data['errors'],
            c=nodeColors, marker='.', linestyle='None')
        errArea = self.axErr.fill_between(data['errR'],
                                          data['errors'],
                                          #  self.errR[ii],
                                          # self.errors[ii],
                                          color='r',
                                          alpha=0.75)

        # Third pane: element sizes
        l0line = self.axL.scatter(data['elemR'],
                                  data['elemL'],
                                  # self.elemR[ii],
                                  #                     self.elemL[ii],
                                  c=BASE, marker='.')

        # plt.tight_layout()
        artists.append(title)
        artists.append(errLine)
        artists.append(simLine[0])
        artists.append(l0line)
        artists.append(errArea)

        # artistSet.append(artists)
        return artists


class CurrentPlot(FigureAnimator):
    def __init__(self, fig, study, fullarrow=False, showInset=True, showAll=False, normalAxis=2, normalCoord=0.):
        super().__init__(fig, study)

        # inf lower bound needed for log colorbar
        self.crange = ScaleRange()
        self.cvals = []
        self.pts = []

        self.inset = None
        if fig is None:
            self.ax = new3dPlot(study.bbox)
            self.dim = 3
        else:
            self.dim = 2
            if showInset:
                self.ax = plt.subplot2grid((3, 3), (0, 1),
                                           colspan=2, rowspan=2,
                                           fig=fig)
                # self.inset=addInset(self.ax,
                #                  3*self.rElec,
                #                  self.study.bbox[3],
                #                  (-.2, -.2))
            else:
                self.ax = fig.add_subplot()

            bnds = study.bbox[[0, 3, 1, 4]]
            formatXYAxis(self.ax, bnds)

        self.rElec = 0
        self.iSrc = []
        self.fullarrow = fullarrow
        self.showInset = showInset
        self.showAll = showAll

        self.normalAxis = normalAxis
        self.normalCoord = normalCoord

        self.dataScales = {
            'iRange': ScaleRange()}

    def addSimulationData(self, sim, append=False):
        # i, E = sim.getEdgeCurrents()

        # coords = sim.mesh.nodeCoords
        # isSrc = sim.nodeRoleTable == 2
        # whichSrc = sim.nodeRoleVals[isSrc]
        # srcCoords = np.array([s.coords for s in sim.currentSources])
        # coords[isSrc] = srcCoords[whichSrc]

        if len(sim.currentSources) > 0:
            rElec = sim.currentSources[0].radius
            self.rElec = rElec

        if len(sim.voltageSources) > 0:
            rElec = sim.voltageSources[0].radius
            self.rElec = rElec

        self.iSrc.append(sim.currentSources[0].value)

        # if self.dim == 3:
        #     eplane = E
        #     iplane = i

        # else:
        #     #only get points in specified plane
        #     centerCoord=np.zeros(3)
        #     centerCoord[self.normalAxis]=self.normalCoord
        #     level=sim.intifyCoords(centerCoord)[self.normalAxis]

        #     if self.showAll:
        #         edgeInPlane=np.ones_like(i,dtype=np.bool_)
        #     else:
        #         ptInPlane = sim.intifyCoords()[:, self.normalAxis] == level
        #         edgeInPlane = [np.all(ptInPlane[e]) for e in E]

        #     eplane = E[edgeInPlane]
        #     iplane = i[edgeInPlane]

        # #reworked projection
        # el,_,meshpt=sim.getElementsInPlane()

        # # discard zero currents
        # isok = iplane > 0
        # iOK = iplane[isok]
        # ptOK = coords[eplane[isok]]

        iOK, ptOK, meshpt = sim.getCurrentsInPlane()
        # _,_,meshpt=sim.getElementsInPlane()

        data = {
            'mesh': meshpt,
            'currents': iOK,
            'pts': ptOK,
            'iSrc': sim.currentSources[0].value
        }

        self.cvals.append(iOK)
        self.pts.append(ptOK)

        self.dataScales['iRange'].update(iOK)

        if append:
            self.dataSets.append(data)

        return data

    def getArtists(self, setnum):
        # include injected currents in scaling
        dscale = self.dataScales['iRange']
        dscale.update(np.array(self.iSrc))

        cmap, cnorm = getCmap(dscale.get(), logscale=True)
        # cmap=mpl.cm.YlOrBr
        # cnorm=mpl.colors.LogNorm(vmin=self.crange[0],
        #                           vmax=self.crange[1])

        if setnum == 0:
            plt.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                         ax=self.ax)
            plt.title('Edge currents')
            inset = None
            if self.showInset & (self.rElec > 0) & (self.dim != 3):
                self.inset = addInset(self.ax,
                                      3*self.rElec,
                                      self.study.bbox[3],
                                      (-.2, -.2))
                showSourceBoundary([self.ax, self.inset],
                                   self.rElec)

        artists = []

        data = self.dataSets[setnum]

        # for data in range(len(self.dataSets)):
        # pt = self.pts[ii]
        # cv = self.cvals[ii]
        pt = data['pts']
        cv = data['currents']
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
            if self.inset is not None:
                axes.append(self.inset)

            for ax in axes:
                edgecol = mpl.collections.LineCollection(data['mesh'],
                                                         color=(0., 0., 0., 0.1))
                art.append(ax.add_collection(edgecol))
                if self.fullarrow:
                    art.append(ax.quiver(x, y, a, b,
                                         color=colors,
                                         angles='xy',
                                         scale_units='xy',
                                         scale=1))  # ,
                    # headlength=15,
                    # headwidth=10))
                else:
                    # TODO: hack to get edges; assumes xy plane at z=0
                    # art.append(showEdges2d(ax,
                    #                        pt[:,:,:-1],
                    #                        edgeColors=(0.,0.,0.),
                    #                        alpha=0.25))
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
                                      color=cmap(cnorm(data['iSrc']))))

            # artists.append(art)

        return art


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
            if any(va > 0):
                self.knee = min(self.knee, min(va[va > 0]))

    def get(self):
        isBipolar = (self.min < 0) & (self.max > 0)
        if isBipolar:
            out = np.array([self.min, self.knee, self.max])
        else:
            out = np.array([self.min, self.max])
        return out


def hideBorders(axis, hidex=False):
    axis.yaxis.set_visible(False)
    [axis.spines[k].set_visible(False) for k in axis.spines]
    if hidex:
        axis.xaxis.set_visible(False)


class SingleSlice(FigureAnimator):
    def __init__(self, fig, study, timevec=[], tdata=None, datasrc='spaceV'):
        self.bnds = study.bbox[[0, 3, 2, 4]]
        self.tdata = tdata
        self.dataSrc = datasrc
        super().__init__(fig, study)
        self.dataScales = {
            'spaceV': ScaleRange(),
            'absErr': ScaleRange(),
            'relErr': ScaleRange()
        }

        self.timevec = timevec

    def setupFigure(self):

        fig, axes = plt.subplots(2, 2,
                                 gridspec_kw={
                                     'height_ratios': [9, 1],
                                     'width_ratios': [19, 1]})
        # gridspec_kw={
        #     'height_ratios': [9,3],
        #     'width_ratios': [9,1]})

        ax = axes[0, 0]
        tax = axes[1, 0]
        cax = axes[0, 1]
        nonax = axes[1, 1]
        fig.delaxes(nonax)

        formatXYAxis(ax, self.bnds)
        self.tbar = TimingBar(fig, tax, self.tdata)

        # plt.tight_layout()

        # tax.set_xlabel('Time [ms]')

        self.ax = ax
        self.cax = cax
        self.tax = tax
        self.axes = [ax, cax, tax]
        self.fig = fig

    def getArtists(self, setnum, data=None):

        if data is None:
            data = self.dataSets[setnum]

        cMap, cNorm = getCmap(self.dataScales[self.dataSrc].get())

        # if setnum==0:
        mapper = mpl.cm.ScalarMappable(norm=cNorm, cmap=cMap)

        if self.dataSrc == 'spaceV' or self.dataSrc == 'absErr':
            cbarFormat = eform('V')
        else:
            cbarFormat = mpl.ticker.PercentFormatter(xmax=1.,
                                                     decimals=2)
            self.cax.set_ylabel('Relative error')
        self.fig.colorbar(mapper, cax=self.cax,
                          format=cbarFormat)
        if setnum == 0:
            plt.tight_layout()

        artists = patchworkImage(self.ax,
                                 data[self.dataSrc], cMap, cNorm,
                                 self.bnds)

        artists.append(showEdges2d(self.ax,
                                   data['meshPts'],
                                   alpha=0.025))
        artists.extend(self.tbar.getArt(self.timevec[setnum],
                                        setnum))

        return artists

    def addSimulationData(self, sim, append=False):
        vest, _ = sim.analyticalEstimate()

        absErr = sim.nodeVoltages-vest[0]
        relErr = absErr/np.abs(vest[0])

        vArrays, _ = sim.getValuesInPlane()
        absArr, _ = sim.getValuesInPlane(data=absErr)

        _, _, edgePts = sim.getElementsInPlane()

        v1d = util.unravelArraySet(vArrays)

        self.dataScales['spaceV'].update(v1d)
        # self.dataScales['relErr'].update(relErr)
        self.dataScales['absErr'].update(absErr)

        data = {
            'spaceV': vArrays,
            # 'relErr':errArrays,
            'absErr': absArr,
            'meshPts': edgePts}

        if append:
            self.dataSets.append(data)

    def animateStudy(self, fname=None, artists=None, fps=30.):
        # animations=[]

        # artists=[]
        # for ii in range(len(self.dataSets)):
        #     art=self.getArtists(ii)
        #     artists.append(art)

        # animation=ArtistAnimation(self.fig, artists,
        #                           interval=1000/fps,
        #                           repeat_delay=2000,
        #                           blit=True)
        # if fname is None:
        #     plt.show()
        # else:
        #     # writer = FFMpegWriter(fps=50)
        #     fpath=os.path.join(
        #         self.study.studyPath, fname+'.mp4')
        #     animation.save(fpath,fps=fps)

        #     # with writer.saving(self.fig,fpath,dpi=None):
        #     #     for ii in range(len(self.dataSets)):
        #     #         self.getArtists(ii)
        #     #         writer.grab_frame()
        #     #         # self.resetFigure()
        #     # animation=None

        animation = super().animateStudy(fname=fname,
                                         artists=artists,
                                         fps=fps)

        return animation
