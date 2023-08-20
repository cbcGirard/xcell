#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization routines for meshes."""

import pyvista as pv
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import matplotlib as mpl
from matplotlib.ticker import EngFormatter as eform

# from matplotlib.animation import ArtistAnimation, FFMpegWriter
import numpy as np

# import numba as nb
# from scipy.interpolate import interp2d
from scipy.sparse import tril
import os
from matplotlib.cm import ScalarMappable as ScalarMap

# import cmasher as cmr

# import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# import pickle
import pandas

from . import util
from . import misc
from . import colors
from .AAnimation import AAnimation, FWriter

MAX_LOG_SPAN = 2


def engineering_ticks(axis, xunit=None, yunit=None):
    """
    Set axis to use engineering (SI) notation.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to format
    xunit : str, optional
        Displayed unit for x axis, by default None
    yunit : str, optional
        Displayed unit for y axis, by default None
    """
    if xunit is not None:
        axis.xaxis.set_major_formatter(eform(xunit, places=0))
    if yunit is not None:
        axis.yaxis.set_major_formatter(eform(yunit, places=0))


# TODO: document better
class TimingBar:
    def __init__(self, figure, axis=None, data=None):
        """
        Create timing bar in figures

        Parameters
        ----------
        figure : :py:class:`matplotlib.figure.Figure`
            Figure containing timing bar
        axis : :py:class:`matplotlib.axes.Axes`
            Axes to place timing bar in. Adds subplot if None (default).
        data : array-like or dict, optional
            Array of y-values, or dict with additional fields., by default
            None
        """
        self.maxtime = np.inf
        self.fig = figure
        self.axes = []
        if axis is None:
            self.axes.append(figure.add_subplot())
        else:
            self.axes.append(axis)

        ax = self.axes[0]

        ax.xaxis.set_major_formatter(eform("s"))

        [ax.spines[k].set_visible(False) for k in ax.spines]

        self.data = data
        if data is not None:
            if "style" in data:
                if data["style"] != "sweep":
                    if data["style"] == "dot":
                        alpha = 0.5
                    else:
                        alpha = 1
                    ax.plot(data["x"], data["y"], color="C0", alpha=alpha)
                ax.set_ylabel(data["ylabel"])
                ax.yaxis.set_major_formatter(eform(data["unit"]))
                ax.set_yticks([0])
                ax.grid(False)
                ax.grid(visible=True, which="major", axis="y")
            else:
                if "shape" in dir(data):
                    ax.set_xlim(min(data), max(data))
                    ax.yaxis.set_visible(False)
        else:
            ax.yaxis.set_visible(False)

    def get_artists(self, time, step=None):
        """
        Get artists for timing bar at given time

        Parameters
        ----------
        time : float
            Time in seconds to generate bar
        step : int, optional
            Index of timestep stored in .data, by default None
            (use value of time)

        Returns
        -------
        list of artists
            List of artists
        """
        ax = self.axes[0]
        art = []

        if step is not None and self.data is not None:
            if self.data["style"] == "sweep":
                art.append(ax.plot(self.data["x"][:step], self.data["y"][:step], color="C0")[0])
            elif self.data["style"] == "rave":
                art.append(
                    ax.plot(
                        self.data["x"][:step],
                        self.data["y"][:step],
                    )[0]
                )
            elif self.data["style"] == "dot":
                art.append(ax.scatter(self.data["x"][step], self.data["y"][step], marker="o", color="C0", s=2.0))

        else:
            art.append(ax.barh(0, time, color="C0")[0])

        return art


def make_discrete_colors(values, legend_stem="n="):
    """
    Generate colors for discrete (categorical) data.

    Parameters
    ----------
    values : array of numeric
        Non-unique values to be color-coded.
    legend_stem : string, optional
        Text preceding category values in legend. The default is 'n='.

    Returns
    -------
    colors : 2d array
        Colors assigned to each input value.
    handles : :py:class:`matplotlib.patches.Patch`
        Dummy objects to generate color legend
        [with legend(handles=legEntry)].

    """
    colVals, colMap = np.unique(values, return_inverse=True)
    ncols = colVals.shape[0]

    if ncols < 11:
        colSet = mpl.cm.get_cmap("tab10").colors
    else:
        colSet = mpl.cm.get_cmap("tab20").colors

    colors = np.array(colSet)[colMap]

    handles = [
        mpl.patches.Patch(color=colSet[ii], label=legend_stem + "%d" % colVals[ii])
        for ii in range(colVals.shape[0])
    ]

    return colors, handles


def discrete_legend(axis, data, legend_stem="n=", **kwargs):
    """
    Color-code data and generate corresponding legend.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to place legend in.
    data : numeric
        Data to be color-coded.
    legend_stem : string, optional
        Legend text preceding values. The default is 'n='.

    Returns
    -------
    colors : array of colors
        Color corresponding to value of each data point.

    """
    colors, handles = make_discrete_colors(data, legend_stem)
    axis.legend(handles=handles, **kwargs)

    return colors


def format_xy_axis(axis, bounds=None, symlog=False, lindist=None, axlabels=False, xlabel="X [m]", ylabel="Y [m]"):
    """
    Set up axis for planar slices of 3d space.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to format.
    bounds : float[4], optional
        Explicit limits for axes, [xmin, xmax, ymin, ymax]. The default is
        None.
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
    axis.set_aspect("equal")
    axis.xaxis.set_major_formatter(eform("m"))
    axis.yaxis.set_major_formatter(eform("m"))

    if axlabels:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    axis.grid(False)
    if bounds is not None:
        xbnd = [bounds[0], bounds[1]]
        ybnd = [bounds[2], bounds[3]]
        axis.set_xlim(xbnd[0], xbnd[1])
        axis.set_ylim(ybnd[0], ybnd[1])

        xbnd.append(0.0)
        ybnd.append(0.0)
        axis.xaxis.set_ticks(xbnd)
        axis.yaxis.set_ticks(ybnd)

    if symlog:
        axis.set_xscale("symlog", linthresh=lindist)
        axis.set_yscale("symlog", linthresh=lindist)


def undraw_axis(axis):
    """
    Remove contents from axis, preserving scale/labels/ticks.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to clear.

    Returns
    -------
    None.

    """
    # keep limits on redraw
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    for col in axis.collections + axis.lines + axis.images:
        col.remove()

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)


def grouped_scatter(fname, x_category, y_category, group_category, df=None, ax=None, **kwargs):
    """
    Plot study summary data from logfile, grouped by category key.

    Parameters
    ----------
    fname : string
        Filename to read from
    x_category : string
        Key for x-axis quantity
    y_category : string
        Key for y-axis quantity
    group_category : string
        Key to group simulations by
    df : pandas dataframe, optional
        Dataset to query, replaced by contents of fname if None (default)
    ax : :py:class:`matplotlib.axes.Axes`, optional
        Axes to plot on, or create new figure if None (default)
    **kwargs : dict, optional
        Arguments to pass to :py:function:`matplotlib.pyplot.plot`
    """
    if df is None:
        df, cats = import_logfile(fname)
    else:
        cats = df.keys()

    dX = df[x_category]
    dY = df[y_category]
    dL = df[group_category]

    catNames = dL.unique()

    if ax is None:
        ax = plt.figure().add_subplot()
    ax.grid(True)
    ax.set_xlabel(x_category)
    ax.set_ylabel(y_category)
    for ii, label in enumerate(catNames):
        sel = dL == label
        ax.loglog(dX[sel], dY[sel], marker=".", label=group_category + ":\n" + str(label), **kwargs)

    plt.tight_layout()


# TODO: merge with study.load_logfile
def import_logfile(fname):
    """
    Import .csv logfile to pandas dataframe

    Parameters
    ----------
    fname : string
        Name of file to read

    Returns
    -------
    df : pandas dataframe
        Imported data
    cats : strings
        Category labels
    """
    df = pandas.read_csv(fname)
    cats = df.keys()

    return df, cats


def plot_study_performance(study, plot_ratios=False, **kwargs):
    """
    Plot breakdown of computational time.

    Parameters
    ----------
    study : `.Study`
        Study to analyize
    plot_ratios : bool, optional
        Whether to plot the ratios of CPU to wall time, by default False

    Returns
    -------
    figs : list of :py:class:`matplotlib.figures.Figure`
        Output figures
    """
    fig1, axes = plt.subplots(2, 1)
    figs = [fig1]
    fn = study.study_path + "/log.csv"

    for ax, ttype in zip(axes, ["Wall", "CPU"]):
        import_and_plot_times(fn, ttype, ax, **kwargs)

    if plot_ratios:
        fig2, _ = import_and_plot_times(fn, time_type="Ratio", ax=None, **kwargs)
        figs.append(fig2)

    return figs


def import_and_plot_times(
    fname,
    time_type="Wall",
    ax=None,
    only_category=None,
    only_value=None,
    x_category="Number of elements",
    **kwargs
):
    """
    Generate plots of step duration from study logfile.

    Parameters
    ----------
    fname : string
        Name of logfile to import from
    time_type : string, optional
        Plot 'Wall' (default) or 'CPU' time
    ax : :py:class:`matplotlib.axes.Axes`, optional
        Axes to plot in, or create new figure if None (default)
    only_category : string, optional
        Category in dataset to filter by, or use all if None (default)
    only_value : any, optional
        Plot only simulations whose only_category matches this value, or
        use all if None (default)
    x_category : string, optional
        Category to use for x-axis values, by default 'Number of elements'
    **kwargs : dict, optional
        Arguments passed to :py:function:`matplotlib.pyplot.plot`

    Returns
    -------
    (fig, ax)
        Figure and axes objects of plot
    """
    df, cats = import_logfile(fname)
    if only_value is not None:
        df = df[df[only_category] == only_value]

    xvals = df[x_category].to_numpy()

    if time_type == "Ratio":
        cwall = [c for c in cats if c.find("Wall") > 0]
        cCPU = [c for c in cats if c.find("CPU") > 0]
        tcpu = df[cCPU].to_numpy().transpose()
        twall = df[cwall].to_numpy().transpose()

        stepTimes = tcpu / twall
        tcols = cCPU

    else:
        tcols = [c for c in cats if c.find(time_type) > 0 and c.find("Total") < 0]

        stepTimes = df[tcols].to_numpy().transpose()
    step_names = [c[: c.find("[") - 1] for c in tcols]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    if time_type == "Ratio":
        for val, lbl in zip(stepTimes, step_names):
            ax.plot(xvals, val, label=lbl, **kwargs)

        ax.set_ylabel("Estimated parallel speedup")
        ax.set_xlabel(x_category)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(eform())

    else:
        _stacked_time_plot(ax, xvals, stepTimes, step_names, **kwargs)
        ax.set_xlabel(x_category)
        ax.set_ylabel(time_type + " time [s]")

    ax.figure.tight_layout()

    return fig, ax


def _stacked_time_plot(axis, xvals, stepTimes, step_names, **kwargs):
    blobs = axis.stackplot(xvals, stepTimes, baseline="zero", labels=step_names, **kwargs)

    if "hatch" in kwargs:
        for b, h in zip(blobs, kwargs["hatch"]):
            b.set_hatch(h)

    axis.xaxis.set_major_formatter(eform())


def outsideLegend(axis=None, reverse_order=False, where="right", **kwargs):
    """
    Create legend outside of axes.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`, optional
        Axes to generate legend, or current axes if None (default)
    reverse_order : bool, optional
        Invert order of legend entries, by default False
    where : str, optional
        Placement of legend, either 'right' (default) or at bottom
    """
    if axis is None:
        axis = plt.gca()

    handles, labels = axis.get_legend_handles_labels()

    if reverse_order:
        h = reversed(handles)
        l = reversed(labels)
    else:
        h = handles
        l = labels

    legargs = {
        "borderaxespad": 0.0,
    }
    if where == "right":
        legargs.update({"bbox_to_anchor": (1.05, 1), "loc": "upper left"})
    else:
        legargs.update(
            {
                "bbox_to_anchor": (0.0, 0.0, 1.0, 0.3),
                "bbox_transform": axis.figure.transFigure,
                "mode": "expand",
                "loc": "lower left",
            }
        )

    legargs.update(kwargs)

    axis.legend(h, l, **legargs)


def show_3d_edges(axis, coords, edge_indices, edge_values=None, colorbar=True, **kwargs):
    """
    Plot 3d view of mesh edges.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to plot within.
    coords : float[:,3]
        Cartesian coordinates of mesh nodes
    edge_indices : int[:,2]
        Array-like where each row contains the two indices of the
        edge's endpoints.
    edge_values : any, optional
        Value assigned to each edge for coloration, e.g.
        conductances or current; by default None
    colorbar : bool, optional
        Whether to generate a colorbar for the edge data,
        by default True

    Returns
    -------
    :py:class:`mpl_toolkits.mplot3d.art3d.Line3DCollection`
        Artist collection for 3d lines
    """
    edgePts = [[coords[a, :], coords[b, :]] for a, b in edge_indices]

    if "colors" in kwargs:
        pass
    else:
        if edge_values is not None:
            (colormap, color_norm) = get_cmap(edge_values)
            gColor = colormap(color_norm(edge_values))
            kwargs["alpha"] = 1.0
            kwargs["colors"] = gColor
            if colorbar:
                axis.figure.colorbar(ScalarMap(norm=color_norm, cmap=colormap))

        else:
            kwargs["colors"] = colors.BASE
            kwargs["alpha"] = 0.05

    gCollection = p3d.art3d.Line3DCollection(edgePts, **kwargs)

    axis.add_collection(gCollection)
    return gCollection


def show_3d_nodes(axis, coords, node_values, colormap=None, color_norm=None, colors=None, **kwargs):
    """
    Show mesh nodes in 3d.

    Parameters
    ----------
    axis : maplotlib Axes
        Axes to plot on.
    coords : float[:,3]
        Cartesian coordinates of points to plot.
    node_values : numeric array
        Values to set node coloring.
    colormap : colormap, optional
        Colormap to use for node coloring.
        The default is None, which uses :py:function:`.get_cmap`.
    color_norm : matplotlib Norm, optional
        Norm to use for node coloring.
        The default is None, which uses :py:function:`.get_cmap`.
    colors : color-like array, optional
        Color to use at each point, overriding colormap.
        The default is None, which sets colors by colormap and norm.

    Returns
    -------
    artists : :py:class:`matplotlib.collections.PathCollection`
        Collection of point artists.

    """

    if axis is None:
        axis = plt.gca()
    if colors is None:
        if colormap is None and color_norm is None:
            (colormap, color_norm) = get_cmap(node_values)
        vColors = colormap(color_norm(node_values))
        axis.figure.colorbar(ScalarMap(norm=color_norm, cmap=colormap))

    elif len(colors) == 0:
        vColors = discrete_legend(axis, node_values)
    else:
        vColors = colors

    x, y, z = np.hsplit(coords, 3)

    artists = axis.scatter3D(x, y, z, c=vColors, **kwargs)
    return artists


def get_cmap(vals, forceBipolar=False, logscale=False):
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
    colormap : colormap
        Selected colormap.
    color_norm : matplotlib Norm
        Norm function to map data to (0,1) range of colormap.

    """
    if type(vals) == ScaleRange:
        knee = vals.knee
        mn = vals.min
        mx = vals.max
    else:
        mn = min(vals)
        mx = max(vals)
        va = abs(vals)
        if any(va > 0):
            knee = min(va[va > 0])
        else:
            knee = 0
        ratio = abs(mn / mx)

    span = mx - mn
    fracPos = abs(mx / span)
    fracNeg = abs(mn / span)

    crosses = (mx * mn) < 0

    if (crosses and (0.05 < fracPos < 0.95)) or forceBipolar:
        # significant data on either side of zero; use symmetric
        amax = max(abs(mn), mx)

        colormap = colors.CM_BIPOLAR
        if logscale:
            color_norm = mpl.colors.SymLogNorm(linthresh=knee, vmin=-amax, vmax=amax)
        else:
            color_norm = mpl.colors.CenteredNorm(halfrange=amax)
    else:
        if fracPos > fracNeg:
            colormap = colors.CM_MONO.copy()
        else:
            colormap = colors.CM_MONO.reversed()

        if logscale:
            color_norm = mpl.colors.LogNorm(vmin=knee, vmax=mx)

        else:
            color_norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    return (colormap, color_norm)


def new3dPlot(bounding_box=None, *args, fig=None):
    """
    Create 3d axes with equal scaling in x, y, and z.

    Parameters
    ----------
    bounding_box : float[6], optional
        Bounding box in xcell ordering (-x,-y,-z,+x,...).
        If None, scale to (-1,+1) (default)
    fig : _type_, optional
        Figure to create axes in, or make new figure if None (default)

    Returns
    -------
    :py:class:`matplotlib.axes.Axes`
        Newly created 3d axes

    Notes
    -----------
    Based on https://stackoverflow.com/a/13701747

    """
    if fig is None:
        fig = plt.figure()

    if bounding_box is None:
        bounding_box = np.array([-1, -1, -1, 1, 1, 1], dtype=np.float64)

    axis = fig.add_subplot(*args, projection="3d")
    axis.set_proj_type("ortho")
    origin = bounding_box[:3]
    span = bounding_box[3:] - origin
    center = 0.5 * (origin + bounding_box[3:])

    max_range = max(span)

    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * center[0]
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * center[1]
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * center[2]
    for xb, yb, zb in zip(Xb, Yb, Zb):
        axis.plot([xb], [yb], [zb], color=[1, 1, 1, 0])

    [a.set_major_formatter(eform("m")) for a in [axis.xaxis, axis.yaxis, axis.zaxis]]

    return axis


def animated_title(figure, text, axis=None, **kwargs):
    """
    Create animation-ready title.

    Parameters
    ----------
    figure : _type_
        Figure to place title in (or assign to axis if None)
    text : string
        Text of title
    axis : _type_, optional
        Axes to title if figure is None, by default None

    Returns
    -------
    :py:class:`matplotlib.text.Text`
        Title artist
    """
    if figure is None:
        dest = axis
        kwargs["transform"] = axis.transAxes
    else:
        dest = figure

    if "vspace" in kwargs:
        d = 1.0 - kwargs.pop("vspace")
    else:
        d = 1.0

    title = dest.text(0.5, d, text, horizontalalignment="center", verticalalignment="top", **kwargs)
    return title


def resample_plane(axis, sim, move_to_center=True, elements=None, data=None):
    """
    Resample simulation data with a uniform planar grid.

    Parameters
    ----------
    axis : :py:class:`matplotlib.axes.Axes`
        Axes to plot within.
    sim : :py:class:`.Simulation`
        Simulation to sample data from.
    move_to_center : bool, default True
        Whether to sample at center of gridpoints instead of gridpoints
        proper. Default true.
    elements : list of :py:class:`xcell.elements.Element`, default None
        Elements to sample from, or use entire mesh if None (default).
    data : numeric array-like, default None
        Value at each mesh point to interpolate from, or use nodal
        voltages if None (default)

    Returns
    -------
    float[:,:]
        2d array of interpolated data for visualization.

    """
    xl, xh = axis.get_xlim()
    yl, yh = axis.get_ylim()

    if move_to_center:
        nX = 512
    else:
        nX = 513

    ppts = util.make_grid_points(513, xh, xl, yh, yl, centers=move_to_center)

    pts = np.hstack((ppts, np.zeros((ppts.shape[0], 1))))

    vInterp = sim.interpolate_at_points(pts, elements, data)

    return vInterp.reshape((nX, nX))


def show_current_vectors(axis, pts, vecs):
    """
    Plot current vectors in 3d.

    Parameters
    ----------
    axis : axis
        Axis to plot on.
    pts : float[:,3]
        Coordinates of vector tails.
    vecs : float[:,3]
        Current density vectors.

    Returns
    -------
    art : Quiver
        DESCRIPTION.

    """
    X, Y, Z = np.hsplit(pts, 3)
    dx, dy, dz = np.hsplit(vecs, 3)

    iMag = np.linalg.norm(vecs, axis=1)

    colormap, color_norm = get_cmap(iMag)

    colors = colormap(iMag)

    art = axis.quiver3D(X, Y, Z, dx, dy, dz, colors=colors)
    return art


def show_2d_edges(axis, edge_points, edge_colors=None, **kwargs):
    """
    Show mesh edges in a 2d plot.

    Parameters
    ----------
    axis : matplotlib axis
        2d axis to plot in.
    edge_points : List of pairs of xy coords
        List of endpoints of each edge.
    edge_colors : array-like of color-like, optional
        List of colors for each edge. The default is None, which uses
        :py:constant:`xcell.colors.FAINT`.
    **kwargs : dict, optional
        Args passed to :py:class:`matplotlib.collections.LineCollection`

    Returns
    -------
    edgeCol : :py:class:`matplotlib.collections.LineCollection`
        Artist for displaying the edges.

    """
    if "colors" not in kwargs:
        if edge_colors is None:
            # kwargs['colors'] = (0., 0., 0.,)

            # kwargs['alpha'] = 0.05
            kwargs["colors"] = colors.FAINT
        else:
            kwargs["colors"] = edge_colors
            # alpha=0.05

    if "linewidths" not in kwargs:
        kwargs["linewidths"] = 0.5

    edgeCol = mpl.collections.LineCollection(edge_points, **kwargs)
    axis.add_collection(edgeCol)
    return edgeCol


def paired_bars(data1, data2, labels, categories=None, axis=None, aright=None):
    """
    Plot paired bars.

    Parameters
    ----------
    data1 : float[:]
        First list of data points
    data2 : float[:]
        Other data set
    labels : str[:]
        Legend labels for the two data sets
    categories : str[:], optional
        X-axis labels, or simply numbered if None (default)
    axis : Axis
        Axes to plot on, or current if None
    aright : Axis
        Paired axis for plotting data2 on right-hand scale.
    """
    if axis is None:
        axis = plt.gca()

    if aright is None:
        aright = axis

    if categories is None:
        categories = np.arange(len(data1))

    barpos = np.arange(len(categories))
    barw = 0.25

    tart = axis.bar(barpos - barw / 2, height=data1, width=barw, color="C0", label=labels[0])

    eart = aright.bar(barpos + barw / 2, height=data2, width=barw, color="C1", label=labels[1])

    axis.set_xticks(barpos)
    axis.set_xticklabels(categories)

    axis.legend(handles=[tart, eart])

    axes = [axis, aright]
    ntick = 0

    nticks = [len(a.get_yticks()) for a in axes]
    ntick = max(nticks)
    for a in axes:
        dtick = a.get_yticks()[1]
        a.set_yticks(np.arange(ntick) * dtick)


def show_source_boundary(axes, radius, source_center=np.zeros(2)):
    """
    Plot faint ring representing source's boundary.

    Parameters
    ----------
    axes : [axis]
        Axes to plot within
    radius : float
        Source radius.
    source_center : float[:], optional
        Center of source. The default is np.zeros(2).

    Returns
    -------
    None.

    """
    tht = np.linspace(0, 2 * np.pi)
    x = radius * np.cos(tht) + source_center[0]
    y = radius * np.sin(tht) + source_center[1]

    for ax in axes:
        ax.plot(x, y, color=colors.BASE)  # , alpha=0.1)


def _add_inset(baseAxis, rInset, xmax, relativeLoc=(0.5, -0.65)):
    """
    Create sub-axis for plotting a zoomed-in view of the main axis.

    Plot commands executed in the main axis DO NOT automatically appear
    in inset; to keep them synchronized, use a pattern like

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
    insetZoom = 0.8 * xmax / rInset
    inset = zoomed_inset_axes(
        baseAxis, zoom=insetZoom, bbox_to_anchor=relativeLoc, loc="center", bbox_transform=baseAxis.transAxes
    )

    inset.set_xlim(-rInset, rInset)
    inset.set_ylim(-rInset, rInset)
    bnds = rInset * np.array([-1, 1, -1, 1])
    format_xy_axis(inset, bnds, xlabel=None, ylabel=None)

    mark_inset(baseAxis, inset, loc1=1, loc2=2, fc="none", ec="0.5")

    return inset


def patchwork_image(axis, masked_arrays, colormap, color_norm, extent):
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
    colormap : colormap
        Desired colormap for image.
    color_norm : norm
        Desired norm for image.

    Returns
    -------
    imlist : list of artists
        Artists required to tile region.

    """
    imlist = []

    cmap = colormap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))

    for vmask in masked_arrays:
        im = axis.imshow(
            vmask, origin="lower", extent=extent, cmap=cmap, norm=color_norm, interpolation="bilinear"
        )
        imlist.append(im)

    return imlist


def show_mesh(setup, axis=None):
    """
    Visualize simulation mesh.

    Parameters
    ----------
    setup : :py:class:`xcell.Simulation`
        Simulation data to plot.
    axis : _type_, optional
        3d axes to plot within, or create new if None (default)

    Returns
    -------
    _type_
        3d axes
    """
    if axis is None:
        ax = new3dPlot(setup.mesh.bbox)
    else:
        ax = axis

    mcoord, medge = setup.get_mesh_geometry()

    show_3d_edges(ax, mcoord, medge)

    return ax


def get_planar_edge_points(coords, edges, normal_axis=2, axis_coordinate=0.0):
    """
    Get edges that lie in the specified plane.

    Parameters
    ----------
    coords : float[:,3]
        Cartesian coordinates of mesh nodes
    edges : int[:,2]
        List of indices of endpoints for each edge
    normal_axis : int, optional
        Axis out of (x,y,z) parallel plane normal, by default 2 (z)
    axis_coordinate : int, optional
        Coordinate of plane along normal, by default 0.0

    Returns
    -------
    List of pairs of float[2]
        Edge points, suitable for
        :py:class:`matplotlib.collections.LineCollection`
    """
    sel = np.equal(coords[:, normal_axis], axis_coordinate)
    selAx = np.array([n for n in range(3) if n != normal_axis])

    planeCoords = coords[:, selAx]

    edgePts = [[planeCoords[a, :], planeCoords[b, :]] for a, b in edges if sel[a] & sel[b]]

    return edgePts


class FigureAnimator:
    """
    Base class for matplotlib animations.

    Attributes
    ----------

    datasets :
        Array of dicts containing plot data.
    data_category : str
        Key for which type of data to use for plot.
    data_scales : dict of `.ScaleRange`
        Scaling information for each data type
    """

    def __init__(self, fig, study, prefs=None):
        """
        Create new animator

        Parameters
        ----------
        fig : _type_
            Figure to animate, or create new figure if None (default)
        study : _type_
            Study managing file IO
        prefs : dict, optional
            Dict of plot preferences, by default None
        """
        if fig is None:
            fig = plt.figure()

        self.fig = fig
        self.study = study
        self.axes = []

        if prefs is None:
            self.prefs = dict()
        else:
            self.prefs = prefs

        self.datasets = []
        self.data_category = []
        self.data_scales = {}

        self.setup_figure()

    def __getstate__(self):
        state = self.__dict__.copy()

        state["fig"] = None
        state["axes"] = []

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.fig = plt.gcf()
        self.axes = self.fig.axes

    def setup_figure(self):
        """Format figure."""
        pass

    def add_simulation_data(self, sim, append=False):
        """
        Extract plottable data from simulation.

        Parameters
        ----------
        sim : _type_
            Simulation to add data from
        append : bool, optional
            Append to internal animation data, by default False
        """
        pass

    def animate_study(self, fname=None, artists=None, extra_artists=None, fps=1.0, vector_frames=[]):
        """_summary_

        Parameters
        ----------
        fname : string, optional
            Name for saved file, or use class name if None (default)
        artists : list of list of artists, optional
            List of artists at each frame, or generate artists from
            self.datasets if None (default)
        extra_artists : list of list of artists, optional
            Artists for additional frames of animation, by default None
        fps : float, optional
            Framerate of animation, by default 1.0
        vector_frames : list, optional
            Indices of frames to save snapshots, by default []

        Returns
        -------
        :py:class:`matplotlib.animation.ArtistAnimation`
            Animated figure
        """
        if artists is None:
            artists = []
            if len(vector_frames) > 0:
                frameList = vector_frames
            else:
                frameList = range(len(self.datasets))
            for ii in frameList:
                artists.append(self.get_artists(ii))
                if extra_artists is not None:
                    artists[-1].extend(extra_artists[ii])

        animation = AAnimation(self.fig, artists, interval=1000 / fps, repeat_delay=2000, blit=False)

        if fname is None:
            plt.show()

        else:
            fstem = os.path.join(self.study.study_path, fname)
            if not os.path.exists(fstem):
                os.makedirs(fstem)
            writer = FWriter(fps=int(fps))
            animation.save(
                fstem + ".mp4", writer=writer, vector_frames=vector_frames, frame_prefix=os.path.join(fstem, fname)
            )

            self.study.save(self, fname, ".adata")

        return animation

    def get_study_data(self, **kwargs):
        """
        Load all data in study.

        Parameters
        ----------
        **kwargs, optional
            Arguments passed to :py:method:`xcell.Study.get_saved_simulations`

        """
        fnames, cats = self.study.get_saved_simulations(**kwargs)
        for names, cat in zip(fnames, cats):
            dset = []
            dcat = []
            for name in names:
                sim = self.study.load_simulation(name)
                self.add_simulation_data(sim, append=True)
                dcat.append(cat)

            self.data_category = dcat

    def get_artists(self, setnumberber=0, data=None):
        """
        Get artists corresponding to given data

        Parameters
        ----------
        setnumberber : int, optional
            Frame from self.datasets to use as data, by default 0
        data : dict, optional
            Data to use instead of self.datasets, by default None

        Returns
        -------
        list of Artists
            Artists for given data
        """
        artists = []
        return artists

    def reset_figure(self):
        """Clear all artists from figure."""
        for axis in self.axes:
            undraw_axis(axis)

    def unify_scales(self, other_scales):
        """
        Update internal data scales with new scales.

        Parameters
        ----------
        other_scales : :py:class:`.ScaleRange`
            New data scales to incorporate
        """
        for key in other_scales.keys():
            self.data_scales[key].update(other_scales[key].get())

    def copy(self, new_figure=None, override_prefs={}):
        """
        Create fresh copy of animator with updated preferences.

        Parameters
        ----------
        new_figure : _type_, optional
            Figure for new animator, or create new figure if None (default)
        override_prefs : dict, optional
            Preferences to override in copy, by default {}

        Returns
        -------
        _type_
            _description_
        """
        newPrefs = self.prefs.copy()
        newPrefs.update(override_prefs)
        newAni = self.__class__(new_figure, self.study, newPrefs)

        params = self.__dict__.copy()
        params["fig"] = newAni.fig
        params["axes"] = newAni.axes

        newAni.__dict__ = params

        return newAni


class SliceSet(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        self.rElec = 0
        if len(study.current_simulation.current_sources) > 0:
            rElec = study.current_simulation.current_sources[0].geometry.radius
            self.rElec = rElec

        if len(study.current_simulation.voltage_sources) > 0:
            rElec = study.current_simulation.voltage_sources[0].geometry.radius
            self.rElec = rElec

        stdPrefs = {
            "showError": True,
            "showInsets": True,
            "relativeError": False,
            "logScale": True,
            "show_nodes": False,
            "fullInterp": True,
        }

        if prefs is not None:
            stdPrefs.update(prefs)

        super().__init__(fig, study, stdPrefs)
        self.data_scales = {"vbounds": ScaleRange(), "errbounds": ScaleRange()}

    def __getstate__(self):
        state = self.__dict__.copy()
        # for a in self.insets:
        #     if a in state['axes']:
        #         state['axes'].remove(a)

        state["insets"] = []
        state["fig"] = None
        state["axes"] = []
        state["grid"] = []

        return state

    def setup_figure(self, resetBounds=False):
        self.bnds = self.study.bbox[[0, 3, 2, 4]]

        if self.prefs["showError"]:
            nplot = 2
            pad = (0.75, 0.15)
            arr = 211
        else:
            nplot = 1
            pad = (0.05, 0.05)
            arr = 111

        self.grid = AxesGrid(
            self.fig,
            arr,  # similar to subplot(144)
            nrows_ncols=(1, nplot),
            axes_pad=pad,
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

        self.grid[0].set_title("Simulated potential [V]")

        if self.prefs["showError"]:
            self.grid[1].set_title("Absolute error [V]")

        insets = []
        if self.rElec > 0 and self.prefs["showError"]:
            for ax in self.grid:
                inset = _add_inset(ax, 3 * self.rElec, self.bnds[1])
                show_source_boundary([ax, inset], self.rElec)
                insets.append(inset)

                self.axes.append(inset)

        self.insets = insets

        for ax in self.grid[:nplot]:
            format_xy_axis(ax, self.bnds)

        if resetBounds:
            self.data_scales["vbounds"] = ScaleRange()
            self.data_scales["errbounds"] = ScaleRange()
            self.rElec = 0

    def add_simulation_data(self, sim, append=False):
        ax = self.axes[0]

        els, _, edgePts = sim.get_elements_in_plane()

        if self.prefs["fullInterp"]:
            vArrays = [resample_plane(ax, sim, elements=els)]
            v1d = vArrays[0].ravel()
        else:
            vArrays, coords = sim.getValuesInPlane(data=None)
            v1d = util.unravel_array_set(vArrays)

        self.data_scales["vbounds"].update(v1d)

        if self.prefs["showError"]:
            # TODO: better filtering to only calculate err where needed
            ana, _ = sim.calculate_analytical_voltage()
            err1d = sim.node_voltages - ana[0]
            if self.prefs["fullInterp"]:
                erArrays = [resample_plane(ax, sim, elements=els, data=err1d)]

            else:
                erArrays, _ = sim.getValuesInPlane(data=err1d)
            self.data_scales["errbounds"].update(err1d)
        else:
            erArrays = None

        # TODO: dehackify so it works with sources other than axes origin

        try:
            inSource = np.zeros(edgePts.shape[0], dtype=bool)
            for src in sim.current_sources:
                inSource |= src.geometry.is_inside(edgePts)
        except:
            inSource = np.linalg.norm(edgePts, axis=2) <= sim.current_sources[0].geometry.radius
        touchesSource = np.logical_xor(inSource[:, 0], inSource[:, 1])
        edge_points = edgePts[~touchesSource]
        sourceEdge = edgePts[touchesSource]

        # quick hack to get nodes in plane directly
        inPlane = sim.mesh.node_coords[:, -1] == 0.0

        pval = sim.node_voltages[inPlane]
        pcoord = sim.mesh.node_coords[inPlane, :-1]

        data = {
            "vArrays": vArrays,
            "errArrays": erArrays,
            "meshPoints": edge_points,
            "sourcePoints": sourceEdge,
            "pvals": pval,
            "pcoords": pcoord,
        }

        if append:
            self.datasets.append(data)

        return data

    def get_artists(self, setnumber, data=None):
        artists = []

        if data is None:
            data = self.datasets[setnumber]

        imAxV = self.axes[0]
        imAxes = [imAxV]

        if self.prefs["showError"]:
            cbarAxV = self.axes[2]
            cbarAxE = self.axes[3]
            imAxE = self.axes[1]
            imAxes.append(imAxE)
            if self.prefs["showInsets"]:
                insets = self.axes[4:]
                imAxes.extend(insets)
        else:
            cbarAxV = self.axes[1]
            if self.prefs["showInsets"]:
                insets = [self.axes[2]]
                imAxes.extend(insets)

        vbounds = self.data_scales["vbounds"]

        vmap, vnorm = get_cmap(vbounds, logscale=self.prefs["logScale"])

        vmappr = ScalarMap(norm=vnorm, cmap=vmap)

        cbarAxV.colorbar(vmappr)

        if self.prefs["showError"]:
            errbounds = self.data_scales["errbounds"]
            emap, enorm = get_cmap(errbounds, forceBipolar=True, logscale=self.prefs["logScale"])
            emappr = ScalarMap(norm=enorm, cmap=emap)
            # vmappr.set_clim((-enorm.halfrange, enorm.halfrange))
            cbarAxE.colorbar(emappr)
            errArt = patchwork_image(
                imAxE,
                # self.maskedErrArr[ii],
                data["errArrays"],
                emap,
                enorm,
                extent=self.bnds,
            )
            artists.extend(errArt)

        vArt = patchwork_image(imAxV, data["vArrays"], vmap, vnorm, extent=self.bnds)

        artists.extend(vArt)

        if self.prefs["show_nodes"]:
            px, py = np.hsplit(data["pcoords"], 2)
            pv = data["pvals"]
            ptart = imAxV.scatter(px, py, c=pv, norm=vnorm, cmap=vmap)

            artists.append(ptart)

        if self.prefs["showInsets"]:
            # insets=self.insets

            artists.extend(
                patchwork_image(
                    insets[0],
                    # self.maskedVArr[ii],
                    data["vArrays"],
                    vmap,
                    vnorm,
                    extent=self.bnds,
                )
            )
            if self.prefs["showError"]:
                artists.extend(
                    patchwork_image(
                        insets[1],
                        # self.maskedErrArr[ii],
                        data["errArrays"],
                        emap,
                        enorm,
                        extent=self.bnds,
                    )
                )

        for ax in imAxes:
            artists.append(show_2d_edges(ax, data["meshPoints"]))
            artists.append(
                show_2d_edges(ax, data["sourcePoints"], edge_colors=(0.5, 0.5, 0.5), alpha=0.25, linestyles=":")
            )

        return artists


class ErrorGraph(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        stdPrefs = {
            "showRelativeError": False,
            "colorNodeConnectivity": False,
            "onlyDoF": False,
            "universalPts": True,
            "infoTitle": False,
            "printScale": 1.0,
            "showLegend": True,
        }
        if prefs is not None:
            stdPrefs.update(prefs)
        super().__init__(fig, study, stdPrefs)

        self.data_scales = {"vsim": ScaleRange(), "error": ScaleRange()}

    def setup_figure(self, labelX=True, labelY=True, newAxes=True):
        """
        Format figure.

        Parameters
        ----------
        labelX : bool, optional
            Show x label, by default True
        labelY : bool, optional
            Show y label, by default True
        newAxes : bool, optional
            Generate new axes, by default True (use self.axes otherwise)
        """
        # TODO: sane way to set figure size, aspect ratio for print vs screen
        if type(self.fig) == mpl.figure.Figure:
            figH = self.fig.get_figheight()
            figW = self.fig.get_figwidth()

            scaler = self.prefs["printScale"]

            if type(scaler) is list:
                kW, kH = scaler
                self.fig.set_figheight(9.0 / kH)
                self.fig.set_figwidth(6.0 / kW)
            else:
                self.fig.set_figheight(1.5 * figH * scaler)
                self.fig.set_figwidth(figW * scaler)

        if newAxes:
            axRatios = [5, 5, 2]

            axes = self.fig.subplots(3, 1, gridspec_kw={"height_ratios": axRatios})
        else:
            axes = self.axes

        axV, axErr, axL = axes

        for a in axes:
            a.grid(True)
            a.set_xscale("log")

        axL.set_yscale("log")

        axErr.sharex(axV)
        axL.sharex(axV)

        hideXticks = [axV, axErr]

        if labelX:
            axL.set_xlabel("Distance from source")
            axL.xaxis.set_major_formatter(eform("m"))
        else:
            hideXticks.append(axL)

        if labelY:
            axV.set_ylabel("Voltage")

            if self.prefs["showRelativeError"]:
                axErr.set_ylabel("Relative error")
            else:
                axErr.set_ylabel("Absolute error")
                axErr.yaxis.set_major_formatter(eform("V"))

            axL.set_ylabel(r"$\ell_0$")

            axV.yaxis.set_major_formatter(eform("V"))
            axL.yaxis.set_major_formatter(eform("m"))
        else:
            [plt.setp(a.get_yticklabels(), visible=False) for a in axes]

        [plt.setp(a.get_xticklabels(), visible=False) for a in hideXticks]

        self.axes = [axV, axErr, axL]

        self.analytic = []
        self.analyticR = []

    def add_simulation_data(self, sim, append=False):
        isDoF = sim.node_role_table == 0
        dofInds = sim.mesh.index_map[isDoF]

        if self.prefs["universalPts"]:
            pts, v = sim.get_universal_points()
            if sim.meshtype == "uniform":
                coords = sim.mesh.node_coords
            else:
                coords = util.indices_to_coordinates(pts, sim.mesh.bbox[:3], sim.mesh.span)
        else:
            v = sim.node_voltages
            coords = sim.mesh.node_coords
            pts = sim.mesh.index_map

        nNodes = len(sim.mesh.node_coords)
        nElems = len(sim.mesh.elements)

        r = np.linalg.norm(coords, axis=1)

        if len(sim.current_sources) > 0:
            rElec = sim.current_sources[0].geometry.radius
            self.rElec = rElec

        if len(sim.voltage_sources) > 0:
            rElec = sim.voltage_sources[0].geometry.radius
            self.rElec = rElec

        lmin = np.log10(rElec / 2)
        lmax = np.log10(max(r))

        if len(self.analyticR) == 0:
            rDense = np.logspace(lmin, lmax, 200)

            vD, _ = sim.calculate_analytical_voltage(rDense)
            vAna = vD[0]
            self.analyticR = rDense
            self.analytic = vAna

            self.data_scales["vsim"].update(vAna)

            self.axes[0].set_xlim(left=rElec / 2, right=2 * max(r))

        ErrSum, err, vAna, sorter, _ = sim.calculate_errors(pts)

        rsort = r[sorter]
        rsort[0] = 10**lmin  # for nicer plotting
        vsort = v[sorter]

        # filter non DoF
        if self.prefs["onlyDoF"]:
            isDoF = np.isin(pts, dofInds)

            filt = isDoF[sorter]
        else:
            filt = np.ones_like(sorter, dtype=bool)

        rFilt = rsort[filt]
        if self.prefs["showRelativeError"]:
            errRel = err / vAna
            errSort = errRel[sorter]
        else:
            errSort = err[sorter]

        errFilt = errSort[filt]

        if self.prefs["infoTitle"]:
            title = "%s:%d nodes, %d elements, error %.2g" % (sim.meshtype, nNodes, nElems, ErrSum)
        else:
            title = ""

        l0 = np.array([e.l0 for e in sim.mesh.elements])
        center = np.array([e.origin + 0.5 * e.span for e in sim.mesh.elements])
        elR = np.linalg.norm(center, axis=1)

        data = {
            "title": title,
            "errR": rFilt,
            "errors": errFilt,
            "simR": rsort,
            "simV": vsort,
            "elemL": l0,
            "elemR": elR,
        }

        self.data_scales["vsim"].update(vsort)
        self.data_scales["error"].update(errFilt)

        if self.prefs["colorNodeConnectivity"]:
            # check how connected nodes are
            connGlobal = sim.count_node_connectivity(True)
            connsort = connGlobal[sorter]
            connFilt = connsort[filt]
            self.rColors.append(connFilt)
            data["rColors"] = connFilt

        if append:
            self.datasets.append(data)
        return data

    def get_artists(self, setnumber, data=None):
        artistSet = []

        axV = self.axes[0]
        axErr = self.axes[1]
        axL = self.axes[2]

        if data is None:
            data = self.datasets[setnumber]

        axV.plot(self.analyticR, self.analytic, c=colors.BASE, label="Analytical")

        vbnd = self.data_scales["vsim"].get()
        axV.set_ylim(vbnd[0], vbnd[-1])

        errbnd = self.data_scales["error"].get()
        axErr.set_ylim(errbnd[0], errbnd[-1])

        if self.prefs["colorNodeConnectivity"]:
            allconn = []
            for d in self.rColors:
                allconn.extend(d)
            connNs = np.unique(allconn)
            toN = np.zeros(max(connNs) + 1, dtype=int)
            toN[connNs] = np.arange(connNs.shape[0])

            mcolors = discrete_legend(axErr, connNs, loc="upper right")

        artists = []
        simLine = axV.plot(data["simR"], data["simV"], c="C0", marker=".", label="Simulated")

        if setnumber == 0 and self.prefs["showLegend"]:
            axV.legend(loc="upper right")

        title = animated_title(self.fig, data["title"])

        if self.prefs["colorNodeConnectivity"]:
            rcol = data["rColors"]
            nconn = toN[rcol]
            nodeColors = mcolors[nconn]
        else:
            nodeColors = "r"

        errLine = axErr.scatter(data["errR"], data["errors"], c=nodeColors, marker=".", linestyle="None")
        errArea = axErr.fill_between(data["errR"], data["errors"], color="r", alpha=0.75)

        # Third pane: element sizes
        l0line = axL.scatter(data["elemR"], data["elemL"], c=colors.BASE, marker=".")

        # fix scaling issue
        self.axes[0].set_xlim(min(self.analyticR))
        if self.prefs["infoTitle"]:
            artists.append(title)
        artists.append(errLine)
        artists.append(simLine[0])
        artists.append(l0line)
        artists.append(errArea)

        return artists


class CurrentPlot(FigureAnimator):
    def __init__(self, fig, study, fullarrow=False, showInset=True, showAll=False, normal_axis=2, normalCoord=0.0):
        super().__init__(fig, study)

        self.prefs["logScale"] = True

        self.prefs = {
            "logScale": True,
            "colorbar": True,
            "title": True,
            "scaleToSource": False,
        }

        # inf lower bound needed for log colorbar
        self.crange = ScaleRange()
        self.cvals = []
        self.pts = []

        self.inset = None
        if fig is None:
            ax = new3dPlot(study.bbox)
            self.dim = 3
        else:
            self.dim = 2
            if showInset:
                ax = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2, fig=fig)

            else:
                ax = fig.add_subplot()

            bnds = study.bbox[[0, 3, 1, 4]]
            format_xy_axis(ax, bnds)

        if len(self.axes) > 0:
            self.axes[0] = ax
        else:
            self.axes.append(ax)

        self.rElec = 0
        self.i_source = []
        self.fullarrow = fullarrow
        self.showInset = showInset
        self.showAll = showAll

        self.normal_axis = normal_axis
        self.normalCoord = normalCoord

        self.data_scales = {"iRange": ScaleRange()}

    def add_simulation_data(self, sim, append=False):
        if len(sim.current_sources) > 0:
            rElec = sim.current_sources[0].geometry.radius
            self.rElec = rElec

        if len(sim.voltage_sources) > 0:
            rElec = sim.voltage_sources[0].geometry.radius
            self.rElec = rElec

        self.i_source.append(sim.current_sources[0].value.get_value_at_time(sim.current_time))

        iOK, ptOK, meshpt = sim.getCurrentsInPlane()

        data = {
            "mesh": meshpt,
            "currents": iOK,
            "pts": ptOK,
            "i_source": sim.current_sources[0].value.get_value_at_time(sim.current_time),
        }

        self.cvals.append(iOK)
        self.pts.append(ptOK)

        self.data_scales["iRange"].update(iOK)

        if append:
            self.datasets.append(data)

        return data

    def get_artists(self, setnumber):
        # include injected currents in scaling
        dscale = self.data_scales["iRange"]

        if self.prefs["scaleToSource"]:
            dscale.update(np.array(self.i_source))

        cmap, cnorm = get_cmap(dscale, logscale=self.prefs["logScale"])
        # cmap=mpl.cm.YlOrBr
        # cnorm=mpl.colors.LogNorm(vmin=self.crange[0],
        #                           vmax=self.crange[1])

        if setnumber == 0:
            if self.prefs["colorbar"]:
                plt.colorbar(ScalarMap(norm=cnorm, cmap=cmap), ax=self.axes[0])

            if self.prefs["title"]:
                plt.title("Edge currents")
            inset = None
            if self.showInset & (self.rElec > 0) & (self.dim != 3):
                self.inset = _add_inset(self.axes[0], 3 * self.rElec, self.study.bbox[3], (-0.2, -0.2))
                show_source_boundary([self.axes[0], self.inset], self.rElec)

        artists = []

        data = self.datasets[setnumber]

        # for data in range(len(self.datasets)):
        # pt = self.pts[ii]
        # cv = self.cvals[ii]
        pt = data["pts"]
        cv = data["currents"]
        ccolors = cmap(cnorm(cv))

        x0 = pt[:, 1, :].squeeze()
        d0 = pt[:, 0, :].squeeze() - x0
        m0 = x0 + 0.5 * d0
        mags = d0 / np.linalg.norm(d0, axis=1, keepdims=True)

        if self.dim == 3:
            x, y, z = np.hsplit(x0, 3)
            a, b, c = np.hsplit(d0, 3)
            m1, m2, m3 = np.hsplit(m0, 3)
            v1, v2, v3 = np.hsplit(mags, 3)

            artists.append([self.axes[0].quiver3D(x, y, z, a, b, c, colors=ccolors)])

        else:
            x, y = np.hsplit(x0, 2)
            a, b = np.hsplit(d0, 2)
            m1, m2 = np.hsplit(m0, 2)
            v1, v2 = np.hsplit(mags, 2)
            axes = [self.axes[0]]
            art = []
            if self.inset is not None:
                axes.append(self.inset)

            for ax in axes:
                # edgecol = mpl.collections.LineCollection(data['mesh'],
                #                                          color=colors.FAINT)
                # art.append(ax.add_collection(edgecol))
                if self.fullarrow:
                    art.append(ax.quiver(x, y, a, b, color=ccolors, angles="xy", scale_units="xy", scale=1))  # ,
                    # headlength=15,
                    # headwidth=10))
                else:
                    # TODO: hack to get edges; assumes xy plane at z=0
                    # art.append(show_2d_edges(ax,
                    #                        pt[:,:,:-1],
                    #                        edge_colors=(0.,0.,0.),
                    #                        alpha=0.25))
                    art.append(
                        ax.quiver(
                            m1,
                            m2,
                            a / 2,
                            b / 2,
                            color=ccolors,
                            pivot="mid",
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                        )
                    )
                    # headlength=15,
                    # headwidth=10))

                art.append(ax.scatter(0, 0, color=cmap(cnorm(data["i_source"]))))

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
            self.min = min(self.min, np.nanmin(newVals))
            self.max = max(self.max, np.nanmax(newVals))

            nonz = va > 0
            if any(nonz):
                minva = np.nanmin(va[nonz])
                self.knee = min(self.knee, minva)

            # TODO: something smarter than hard-coding
            fracmax = np.nanmax(np.abs(self.get())) / (10**MAX_LOG_SPAN)
            fracmax = 10 ** (np.rint(np.log10(fracmax)))

            if self.knee < fracmax:
                self.knee = fracmax

    def get(self, forceBipolar=False, forceSymmetric=False):
        isBipolar = (self.min < 0) & (self.max > 0)
        # if isBipolar or forceBipolar:
        #     out = np.array([self.min, self.knee, self.max])
        # else:
        #     out = np.array([self.min, self.max])
        if forceSymmetric:
            maxamp = max(abs(self.min), abs(self.max))
            out = np.array([-maxamp, maxamp])
        else:
            out = np.array([self.min, self.knee, self.max])
        return out


def hideBorders(axis, hidex=False):
    axis.yaxis.set_visible(False)
    [axis.spines[k].set_visible(False) for k in axis.spines]
    if hidex:
        axis.xaxis.set_visible(False)


class SingleSlice(FigureAnimator):
    def __init__(self, fig, study, timevec=[], tdata=None, datasrc="spaceV", prefs=None):
        self.bnds = study.bbox[[0, 3, 2, 4]]
        self.tdata = tdata
        self.dataSrc = datasrc
        self.timevec = timevec

        if prefs is None:
            prefs = {"colorbar": True, "barRatio": [9, 1], "labelAxes": True}
        super().__init__(fig, study, prefs=prefs)
        self.data_scales = {
            "spaceV": ScaleRange(),
            "absErr": ScaleRange(),
            "relErr": ScaleRange(),
            "vAna": ScaleRange(),
        }

    def copy(self, newPrefs=None):
        params = self.__dict__.copy()
        oldPrefs = params["prefs"]

        if newPrefs is not None:
            oldPrefs.update(newPrefs)
        newAni = self.__class__(
            None, self.study, timevec=self.timevec, tdata=self.tdata, datasrc=self.dataSrc, prefs=oldPrefs
        )

        params["fig"] = newAni.fig
        params["axes"] = newAni.axes
        params["tbar"] = newAni.tbar

        newAni.__dict__ = params

        return newAni

    def setup_figure(self):
        ratios = self.prefs["barRatio"]

        if self.prefs["colorbar"]:
            nCol = 2
            widths = self.fig.get_figwidth() * np.array(ratios) / sum(ratios)
            vratios = [widths[0], (1.1 * self.fig.get_figheight() - widths[0])]
        else:
            nCol = 1
            vratios = ratios
        # if self.prefs['colorbar']:
        #     nCol=2
        # else:
        #     nCol=1

        axes = self.fig.subplots(2, nCol, gridspec_kw={"height_ratios": vratios, "width_ratios": ratios[:nCol]})

        if self.prefs["colorbar"]:
            ax = axes[0, 0]
            tax = axes[1, 0]
            cax = axes[0, 1]
            nonax = axes[1, 1]
            self.fig.delaxes(nonax)
        else:
            cax = None
            ax = axes[0]
            tax = axes[1]

        format_xy_axis(ax, self.bnds, axlabels=self.prefs["labelAxes"])
        self.tbar = TimingBar(self.fig, tax, self.tdata)
        tax.set_xlim(self.timevec[0], self.timevec[-1])

        self.axes = [ax, cax, tax]

    def get_artists(self, setnumber, data=None):
        if data is None:
            data = self.datasets[setnumber]

        if data[self.dataSrc] is None:
            artists = []
            meshAlpha = 1.0
            if self.axes[1] is not None:
                self.fig.delaxes(self.axes[1])
                self.axes[1] = None
                plt.tight_layout()
        else:
            meshAlpha = colors.MESH_ALPHA
            colormap, color_norm = get_cmap(self.data_scales[self.dataSrc], logscale=True)

            if setnumber == 0 and self.prefs["colorbar"]:
                mapper = ScalarMap(norm=color_norm, cmap=colormap)

                if self.dataSrc == "spaceV" or self.dataSrc == "absErr" or self.dataSrc == "vAna":
                    cbarFormat = eform("V")
                else:
                    cbarFormat = mpl.ticker.PercentFormatter(xmax=1.0, decimals=2)
                    self.axes[1].set_ylabel("Relative error")
                self.fig.colorbar(mapper, ax=self.axes[0], cax=self.axes[1], format=cbarFormat)
                # if setnumber == 0:
                plt.tight_layout()

            artists = patchwork_image(self.axes[0], data[self.dataSrc], colormap, color_norm, self.bnds)

        artists.append(show_2d_edges(self.axes[0], data["meshPts"], alpha=meshAlpha))
        if self.tbar is not None:
            artists.extend(self.tbar.get_artists(self.timevec[setnumber], setnumber))

        return artists

    def add_simulation_data(self, sim, append=False):
        els, _, edgePts = sim.get_elements_in_plane()

        if sim.node_voltages.shape[0] == 0:
            vArrays = None
            absErr = None
        else:
            vest, _ = sim.calculate_analytical_voltage()

            vAna = np.sum(vest, axis=0)

            absErr = sim.node_voltages - vAna

            vArrays = [resample_plane(self.axes[0], sim, elements=els)]
            v1d = vArrays[0].ravel()

            self.data_scales["spaceV"].update(v1d)
            # self.data_scales['relErr'].update(relErr)
            self.data_scales["absErr"].update(absErr)
            self.data_scales["vAna"].update(vAna)

        data = {"meshPts": edgePts}

        if self.dataSrc == "spaceV":
            data["spaceV"] = vArrays

        if self.dataSrc == "absErr":
            data["absErr"] = [resample_plane(self.axes[0], sim, elements=els, data=absErr)]

        if self.dataSrc == "vAna":
            data["vAna"] = [resample_plane(self.axes[0], sim, elements=els, data=absErr)]

        if append:
            self.datasets.append(data)

    def animate_study(self, fname=None, artists=None, fps=30.0, vector_frames=[], unitStr=None):
        animation = super().animate_study(fname=fname, artists=artists, fps=fps, vector_frames=vector_frames)

        if self.prefs["colorbar"]:
            self.solobar(fname=fname, unit=unitStr)

        return animation

    def solobar(self, fname, unit=None):
        data = self.data_scales[self.dataSrc]

        fbar = makeSoloColorbar(data, unit=unit)

        self.study.save_plot(fbar, fname + "-colorbar")
        plt.close(fbar)


def makeSoloColorbar(data, cmap=None, norm=None, unit=None, **kwargs):
    fbar, printbar = plt.subplots(figsize=[5.0, 0.75])

    if cmap is None and norm is None:
        cmap, norm = get_cmap(data, logscale=True)

    fbar.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=printbar, orientation="horizontal", **kwargs)
    engineering_ticks(printbar, xunit=unit)

    return fbar


class LogError(FigureAnimator):
    def __init__(self, fig, study, prefs=None):
        super().__init__(fig, study)
        if len(self.fig.axes) == 0:
            ax = self.fig.add_subplot()
            self.axes.append(ax)

            ax.xaxis.set_major_formatter(eform("m"))
            ax.yaxis.set_major_formatter(eform("V"))

    def add_simulation_data(self, sim, append=False):
        err1d, err, ana, sr, r = sim.calculate_errors()

        FVU = misc.calculate_fvu(ana, err)
        rmin = min([cs.geometry.radius for cs in sim.current_sources])
        r[r < rmin] = rmin / 2

        nsrc = util.fastcount(sim.node_role_table == 2)

        rAna = np.array([rmin / 2, rmin, max(r)])
        absAna = abs(ana)

        vAna = np.array([max(absAna), max(absAna), min(absAna)])

        data = {
            "r": r[sr],
            "ana": vAna,
            "rAna": rAna,
            "err": abs(err[sr]),
            "err1d": err1d,
            "FVU": FVU,
            "srcPts": nsrc,
        }

        if append:
            self.datasets.append(data)

    def get_artists(self, setnumber, data=None):
        if data is None:
            data = self.datasets[setnumber]

        artists = []
        ax = self.fig.axes[0]

        er1 = ax.loglog(data["rAna"], data["ana"], label="Analytic", color=colors.BASE)
        er2 = ax.loglog(data["r"], data["err"], label="Error", color="r")

        titlestr = "FVU=%.2g, int1=%.2g, %d points in source" % (data["FVU"], data["err1d"], data["srcPts"])

        title = animated_title(None, titlestr, ax)

        if setnumber == 0:
            self.axes[0].legend(loc="upper right")
            # ax.text()

        artists.extend(er1)
        artists.extend(er2)
        artists.append(title)

        return artists


class PVScene(pv.Plotter):
    # for future ref:
    # pdf seems to be fixed at 72 dpi
    def __init__(self, study=None, time=None, **kwargs):
        dpi = None
        hfrac = 0.1
        figsize = None
        if "figsize" in kwargs:
            figsize = kwargs.pop("figsize")
            if "dpi" in kwargs:
                dpi = kwargs.pop("dpi")
                winsize = dpi * np.array(figsize)
                hfrac = 0.5 / figsize[1]
                kwargs["window_size"] = winsize.astype(int)
                figsize[1] *= 1 + hfrac
                if "line_width" in kwargs:
                    lw = kwargs.pop("line_width")
                else:
                    lw = 1.0
                pv.global_theme.line_width = lw * dpi / 72

        super().__init__(**kwargs)
        if time is None:
            f = None
            ax = None
            self.tbar = None
        else:
            f = plt.figure(dpi=dpi, figsize=figsize)
            ax = f.add_axes((0.1, 0.5, 0.8, 0.5))
            self.tbar = TimingBar(figure=f, axis=ax, data=time)
            self.pvchart = pv.ChartMPL(f, size=(1.0, hfrac), loc=(0, 0))
            self.add_chart(self.pvchart)

        self.f = f
        self.ax = ax
        self.study = study
        self.edgeMesh = None

    def setTime(self, time):
        if self.tbar is not None:
            self.tbar.get_artists(time)

    def setup(self, regions, mesh=None, simData=None, **meshkwargs):
        sigargs = {"opacity": 0.5}
        for msh in regions["Conductors"]:
            if mesh is None:
                sigargs.update(meshkwargs)

            self.add_mesh(msh, scalars="sigma", **sigargs)
            # show_scalar_bar=False)
        # hiding scalar bar in mesh add removes color-coding of regions
        if simData != "sigma":
            self.remove_scalar_bar()

        for msh in regions["Insulators"]:
            self.add_mesh(msh, color=colors.BASE, opacity=0.5)

        for msh in regions["Electrodes"]:
            self.add_mesh(msh, color="gold")

        # if mesh is not None and simData is not None:
        #     if 'symlog' in meshkwargs:
        #         meshkwargs.pop('symlog')
        #         self.add_symlog(mesh, **meshkwargs)
        #     else:
        #         self.add_mesh(mesh, scalars=simData, **meshkwargs)

        if mesh is not None:
            if simData is not None:
                meshkwargs["scalars"] = simData

            self.add_mesh(mesh, **meshkwargs)

    def planeview(self, planeBox, scale=1.0, normal="z"):
        self.enable_parallel_projection()

        # position, focus, and up vector, I think?
        self.camera_position = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0)]
        spans = np.diff(planeBox)
        self.camera.parallel_scale = scale * max(spans) / 2

        if len(planeBox) == 6:
            self.camera.focal_point = 0.5 * (planeBox[::2] + planeBox[1::2])

    def clear(self):
        super().clear()
        plt.close(self.f)

    def close(self, **kwargs):
        plt.close(self.f)
        super().close(**kwargs)

    def add_symlog(self, mesh, data=None, **kwargs):
        if "scalars" not in kwargs:
            scalars = mesh.active_scalars_name
            kwargs["scalars"] = scalars
        else:
            scalars = kwargs["scalars"]

        if data is None:
            data = mesh[scalars]

        if "scalar_range" not in kwargs:
            vrange = np.array([min(data), max(data)])
        else:
            vrange = kwargs.pop("scalar_range")

        vspan = np.sign(vrange) * max(np.abs(vrange))

        # constrain
        vspan[0] = vspan[1] / 10 ** (MAX_LOG_SPAN)

        kwargs["show_scalar_bar"] = False
        pvBG = pv.Color(colors.BG, opacity=0)
        bgHSV = mpl.colors.rgb_to_hsv(pvBG.float_rgb)

        lutargs = {
            "log_scale": True,
            "scalar_range": vspan,
            "below_range_color": pvBG,
            "alpha_range": (0.0, 1.0),
            "value_range": (bgHSV[2], 1.0),
        }

        # if 'show_edges' in kwargs:
        #     edge = kwargs.pop('show_edges')
        #     if edge:
        #         self.show_edges(mesh)

        negmesh = mesh.copy()
        negmesh[scalars] = -mesh[scalars]

        neghue = 2 / 3
        if bgHSV[1] == 0:
            poshue = 0.0
            bgHSV[0] = 1 / 3
            lutPos = pv.LookupTable(hue_range=(0, 0), **lutargs)
            lutNeg = pv.LookupTable(hue_range=(2 / 3, 2 / 3), **lutargs)

        else:
            posrange = bgHSV[0]
            poshue = 1.0

            lutPos = pv.LookupTable(hue_range=(bgHSV[0], poshue), **lutargs)
            lutNeg = pv.LookupTable(hue_range=(bgHSV[0], neghue), **lutargs)

        kwargs.pop("cmap")
        super().add_mesh(mesh, cmap=lutPos, **kwargs)
        super().add_mesh(negmesh, cmap=lutNeg, **kwargs)

        # super().add_mesh(mesh, cmap='Reds', opacity='geom', **kwargs)
        # super().add_mesh(negmesh, cmap='Blues', opacity='geom', **kwargs)

    def show_edges(self, mesh):
        self.edgeMesh = mesh
        super().add_mesh(mesh, style="wireframe", opacity=0.2, edge_color=colors.BASE, color=colors.BASE)

    def add_mesh(self, mesh, **kwargs):
        if "show_edges" in kwargs:
            edges = kwargs.pop("show_edges")
            if edges:
                self.show_edges(mesh.copy())

        if "symlog" in kwargs:
            symlog = kwargs.pop("symlog")

            if symlog:
                if "clim" in kwargs:
                    clim = kwargs.pop("clim")
                    kwargs["scalar_range"] = clim
                self.add_symlog(mesh, **kwargs)
                return

        super().add_mesh(mesh, **kwargs)

    def save_graphic(self, filename, **kwargs):
        if self.study is not None:
            fname = os.path.join(self.study.study_path, filename)
        else:
            fname = filename
        super().save_graphic(filename, **kwargs)


class Symlogizer:
    def __init__(self, valRange, linrange=0.1):
        self.crange = np.max(np.abs(valRange)) * np.array([-1, 1])
        self.linrange = linrange

    def change_data(self, newData):
        knee = self.crange[1] / 10 ** (MAX_LOG_SPAN)
        poslog = np.greater_equal(newData, knee)
        neglog = np.less_equal(newData, -knee)

        logspans = 0.5 * (1 - self.linrange)

        xformed = np.empty_like(newData)

        linpts = np.logical_not(np.logical_or(poslog, neglog))

        xformed[poslog] = 1 - np.log(newData[poslog] / self.crange[1]) * logspans / np.log(knee)
        xformed[neglog] = np.log(newData[neglog] / self.crange[0]) * logspans / np.log(knee)
        xformed[linpts] = logspans + self.linrange * (newData[linpts] + knee) / (2 * knee)

        return xformed
