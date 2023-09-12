#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-level helpers."""

import numpy as np
import numba as nb
import scipy
import time
import matplotlib.ticker as tickr

import psutil


# Atrocious typing hacks to force use of uint64
#: Absolute maximum recursion depth to avoid integer overflow
MAXDEPTH = np.array(20, dtype=np.uint64)[()]

#: Number of points along axis at absolute maximum subdivision
MAXPT = np.array(2 ** (MAXDEPTH + 1) + 1, dtype=np.uint64)[()]


def is_scalar(var):
    """Verify var is not a list/array"""
    return not hasattr(var,"__len__")

def point_current_source_voltage(eval_coords, i_source, sigma=1.0, source_location=np.zeros(3, dtype=np.float64)):
    """
    Calculate field from point current source.

    Parameters
    ----------
    eval_coords : float[:,3]
        Points at which to evaluate field.
    i_source : float
        Source current in amperes.
    sigma : float, optional
        Local conductivity in S/m. The default is 1..
    source_location : float[3], optional
        Center of source. The default is np.zeros(3, dtype=np.float64).

    Returns
    -------
    v : float[:]
        Potential at specified points.

    """
    dif = eval_coords - source_location
    k = i_source / (4 * np.pi * sigma)
    v = np.array([k / np.linalg.norm(d) for d in dif])
    return v


def disk_current_source_voltage(eval_coords, i_source, sigma=1.0, source_location=np.zeros(3, dtype=np.float64)):
    """
    Calculate field from a disk current source on planar insulator.

    Parameters
    ----------
    eval_coords : float[:,3]
        Points at which to evaluate field.
    i_source : float
        Source current in amperes.
    sigma : float, optional
        Local conductivity in S/m. The default is 1.0.
    source_location : float[3], optional
        Cartesian coordinates of source center.
        The default is np.zeros(3, dtype=np.float64).

    Returns
    -------
    v : float[:]
        Potential at specified points.

    Notes
    ------
    Analytic expression taken from [1]_.

    References
    -----------
    .. [1] J. Newman, “Resistance for Flow of Current to a Disk,”
       J. Electrochem. Soc., vol. 113, no. 5, p. 501, May 1966,
       doi: 10.1149/1.2424003.


    """
    dif = eval_coords - source_location
    k = i_source / (4 * sigma)
    v = np.array([k / np.linalg.norm(d) for d in dif])
    return v


def round_to_digit(x):
    """
    Round to one significant digit.

    Parameters
    ----------
    x : float
        number to round.

    Returns
    -------
    float
        Rounded number.

    """
    factor = 10 ** np.floor(np.log10(x))

    return factor * np.ceil(x / factor)


def logfloor(val):
    """
    Round down to power of 10.

    Parameters
    ----------
    val : float or float array
        Value(s) to round

    Returns
    -------
    same as val
        Rounded value(s)
    """
    return 10 ** np.floor(np.log10(val))


def logceil(val):
    """
    Round up to power of 10.

    Parameters
    ----------
    val : float or float array
        Value(s) to round

    Returns
    -------
    same as val
        Rounded value(s)
    """
    return 10 ** np.ceil(np.log10(val))


def loground(axis, which="both"):
    """
    Set plot axes' bounds to powers of 10.

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        Plot axes to alter.
    which : str, optional
        Which axes to change bounds: 'x', 'y', or 'both' (default).
    """
    lims = [[logfloor(aa[0]), logceil(aa[1])] for aa in axis.dataLim.get_points().transpose()]  # [xl, yl]]
    if which == "x" or which == "both":
        axis.set_xlim(lims[0])
    if which == "y" or which == "both":
        axis.set_ylim(lims[1])


def make_grid_points(nx, xmax, xmin=None, ymax=None, ymin=None, centers=False):
    """
    Convenience function to make points on xy grid

    Parameters
    ----------
    nx : int
        Number of points per axis.
    xmax : float
        Largest x coordinate
    xmin : float, optional
        Smallest x coordinate, or use -xmax if None (default)
    ymax : float, optional
        Largest y coordinate, or use xmax if None (default)
    ymin : float, optional
        Smallest y coordinate, or use -xmax if None (default)
    centers : bool, optional
        Return the center points of the grid instead of the
        corner vertices, by default False

    Returns
    -------
    float[:,2]
        XY coordinates of points on grid.
    """
    if xmin is None:
        xmin = -xmax
    if ymax is None:
        ymax = xmax
    if ymin is None:
        ymin = -ymax

    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, nx)

    if centers:
        xx = 0.5 * (xx[:-1] + xx[1:])
        yy = 0.5 * (yy[:-1] + yy[1:])

    XX, YY = np.meshgrid(xx, yy)

    pts = np.vstack((XX.ravel(), YY.ravel())).transpose()
    return pts


@nb.njit()
def get_index_dict(sparse_indices):
    """
    Get a dict of subsetIndex: globalIndex.

    Parameters
    ----------
    global_mask : bool array
        Global elements included in subset.

    Returns
    -------
    index_dict : dict
        Dictionary of subset:global indices.

    """
    index_dict = {}
    for ii in range(sparse_indices.shape[0]):
        index_dict[sparse_indices[ii]] = ii

    return index_dict


# TODO: unify dictionary usage
def get_py_dict(sparse_indices):
    """
    Get dictionary mapping a sparse index number to its position in the list of indices.

    Parameters
    ----------
    sparse_indices : int[]
        DESCRIPTION.

    Returns
    -------
    dic : dict
        Dictionary for A[ii]=n such that dic[n]=ii.

    """
    dic = {}
    for ii, v in enumerate(sparse_indices):
        dic[v] = ii
    return dic


@nb.njit(parallel=True)
# @nb.vectorize([nb.int64(nb.int64, nb.int64)])
def renumber_indices(sparse_indices, dense_list):
    """
    Renumber indices according to a subset

    Parameters
    ----------
    sparse_indices : int array
        Indices to renumber
    dense_list : int[:]
        1d array of indices to generate order of

    Returns
    -------
    int array
        Original array with each value replaced by it's position in dense_list
    """
    renumbered = np.empty_like(sparse_indices, dtype=np.uint64)
    dic = get_index_dict(dense_list)

    if sparse_indices.ndim == 1:
        for ii in nb.prange(sparse_indices.shape[0]):
            renumbered[ii] = dic[sparse_indices[ii]]
    else:
        for ii in nb.prange(sparse_indices.shape[0]):
            for jj in nb.prange(sparse_indices.shape[1]):
                renumbered[ii, jj] = dic[sparse_indices[ii, jj]]

    return renumbered


@nb.njit()
def reindex(sparse_value, dense_list):
    """
    Get position of sparse_value in dense_list, returning as soon as found.

    Parameters
    ----------
    sparse_value : int64
        Value to find index of match.
    dense_list : int64[:]
        List of nonconsecutive indices.

    Raises
    ------
    ValueError
        Error if sparse_value not found.

    Returns
    -------
    int64
        index where sparse_value occurs in dense_list.

    """
    for n, val in enumerate(dense_list):
        if val == sparse_value:
            return n

    raise ValueError("not a member of dense_list")


@nb.njit
def minOver(target, vals):
    sel = vals >= target
    return min(vals[sel])


@nb.njit
def maxUnder(target, vals):
    sel = vals <= target
    return max(vals[sel])


@nb.njit()  # ,parallel=True)
# @nb.njit(['int64[:](int64, int64)', 'int64[:](int64, Omitted(int64))'])
def to_bit_array(val):  # , nBits=3):
    """_summary_

    Parameters
    ----------
    val : int
        Integer to convert

    Returns
    -------
    bool[3]
        Array of lower 3 bits from input
    """
    return np.array([(val >> n) & 1 for n in range(3)])


#: Bit representations of ints 0:8
OCT_INDEX_BITS = np.array([to_bit_array(ii) for ii in range(8)])


@nb.njit()
def from_bit_array(arr):
    """
    Calculate integer value of boolean array

    Parameters
    ----------
    arr : bool[:,3]
        Array x,y,z bit values

    Returns
    -------
    uint64
        Integer value of bit array
    """
    val = 0
    nbit = arr.shape[0]
    for ii in nb.prange(nbit):
        val += arr[ii] * 2 ** (nbit - ii - 1)
    return val


@nb.njit
def any_match(search_array, search_values):
    """
    Rapid search if any matches occur
    (returns immediately at first match).

    Parameters
    ----------
    search_array : array
        Array to seach.
    search_values : array
        Values to search array for.

    Returns
    -------
    bool
        Whether match is found.

    """
    for el in search_array.ravel():
        if any(np.isin(search_values, el)):
            return True

    return False


@nb.njit()
def index_to_xyz(ndx, nX):
    """
    Convert scalar index to [x,y,z] indices.

    Parameters
    ----------
    ndx : uint64
        Index to convert.
    nX : uint64
        Number of points per axis.

    Returns
    -------
    int64[:]
        Integer position along [x,y,z] axes.

    """
    arr = np.empty(3, dtype=np.uint64)
    for ii in range(3):
        a, b = divmod(ndx, nX)
        arr[ii] = b
        ndx = a

    return arr


@nb.njit()
def position_to_index(pos, nX):
    """
    Convert [x,y,z] indices to a scalar index

    Parameters
    ----------
    pos : int64[:]
        [x,y,z] indices.
    nX : int64
        Number of points per axis.

    Returns
    -------
    newNdx : int64
        Scalar index equivalent to [x,y,z] triple.

    """
    vals = np.array([nX**n for n in range(3)], dtype=np.uint64)

    newNdx = intdot(pos, vals)

    return newNdx


@nb.njit()
def reduce_functions(l0_function, ref_pts, element_bbox, coefs=None, return_under=True):
    """
    Determine which point-coefficent pairs are not satisfied by the element size.

    Parameters
    ----------
    l0_function : function
        Target element sizes as a function of distance from reference points and their coefficent
    ref_pts : float[:,3]
        Cartesian coordinates of reference points
    element_bbox : float[6]
        Bounding box in order (-x,-y,-z, x, y, z)
    coefs : float[:], optional
        Coefficients per reference point, by default None (Unity coefficent for all points)
    return_under : bool, optional
        Whether to return targets smaller than element (True, default) for splitting,
        or those larger for pruning (False)

    Returns
    -------
    bool[:]
        Which point/coefficent pairs are not satisfied by current element size.
        These should be passed to the child elements, and the rest assumed to be
        satisfied by the children as well.
    """
    nFun = ref_pts.shape[0]

    l0s = l0_function(element_bbox, ref_pts, coefs)

    actual_l0 = np.prod(element_bbox[3:] - element_bbox[:3]) ** (1 / 3)

    which_pts = np.logical_xor(l0s > actual_l0, return_under)

    return which_pts


@nb.njit()  # ,parallel=True)
def intdot(a, b):
    """
    Dot product, guaranteed to maintain uint64 at all times.
    (since np.dot may silently cast to float)

    Parameters
    ----------
    a : int[:,:]
        First part of dot product
    b : int[:]
        Second part of dot product

    Returns
    -------
    uint64[:]
        Integer dot product of operands
    """
    dot = np.empty(a.shape[0], dtype=np.uint64)

    for ii in nb.prange(a.shape[0]):
        dot[ii] = np.sum(a[ii] * b)

    return dot


# TODO: set relative_position default to fem.HEX_POINT_INDICES
@nb.njit()
def get_indices_of_octant(parent_list, relative_position):
    """
    Calculate universal indices of element from list of parents.

    Parameters
    ----------
    parent_list : int8[:]
        List of parent octants' indices.
    relative_position : int8[:,3]
        Offsets of points to return indices for, relative to
        root node of octant.

    Returns
    -------
    uint64[:]
        Universal indices of node
    """
    absPos = calc_xyz_within_octant(parent_list, relative_position)

    indices = position_to_index(absPos, MAXPT)

    return indices


@nb.njit()  # parallel=True, fastmath=True) # 10x slower when parallel...
def calc_xyz_within_octant(parent_list, relative_position):
    """
    Calculate the integer xyz triple for points within octant.

    Parameters
    ----------
    parent_list : int8[:]
        List of parent octants' indices.
    relative_position : int8[:,3]
        Offsets of points to return indices for, relative to
        root node of octant.

    Returns
    -------
    uint64[:]
        Universal indices of node
    """
    origin = octant_list_to_xyz(parent_list)

    npt = relative_position.shape[0]
    depth = MAXDEPTH - parent_list.shape[0]
    scale = 1 << depth

    absPos = np.empty_like(relative_position, dtype=np.uint64)
    for ii in nb.prange(npt):
        for jj in nb.prange(3):
            absPos[ii, jj] = origin[jj] + scale * relative_position[ii, jj]

    return absPos


@nb.njit(
    parallel=True,
)
def indices_to_coordinates(indices, origin, span):
    """
    Convert universal indices to Cartesian coordinates

    Parameters
    ----------
    indices : uint64[:]
        Universal indices to convert.
    origin : float[3]
        Cartesian coordinates of mesh's xyz minimum.
    span : float[3]
        Length of mesh along xyz coordinates.

    Returns
    -------
    coords : float[:,3]
        Cartesian coordinates of points.

    """

    nPt = indices.shape[0]
    coords = np.empty((nPt, 3), dtype=np.float64)
    for ii in nb.prange(nPt):
        ijk = index_to_xyz(indices[ii], MAXPT)
        coords[ii] = span * ijk / (MAXPT - 1) + origin

    return coords


@nb.njit()  # parallel=True) #2x slower if parallel
def octant_list_to_xyz(octant_list):
    """
    Get xyz indices at octant origin.

    Parameters
    ----------
    octant_list : int8[:]
        Indices of each parent octant within its parent.

    Returns
    -------
    XYZ : uint64[3]
        XYZ triple of smallest octant node.

    """
    depth = octant_list.shape[0]
    xyz = np.zeros(3, dtype=np.uint64)
    for ii in nb.prange(depth):
        scale = 1 << (MAXDEPTH - ii)
        ind = octant_list[ii]
        for ax in nb.prange(3):
            if (ind >> ax) & 1:
                xyz[ax] += scale

    return xyz


def get_octant_neighbor_lists(own_index_list):
    """
    Calculate the lists of indices for each neighbor of an octant.

    Parameters
    ----------
    own_index_list : int[:]
        List of parent octants within their parents.

    Returns
    -------
    neighbor_lists : list of int
        List of parent indices for the 6 neighboring octants.

    """
    ownDepth = own_index_list.shape[0]
    nXYZ = reverse_octant_list_to_xyz(own_index_list)
    neighbor_lists = []
    # keep Numba type inference happy

    for nn in nb.prange(6):
        axis = nn // 2
        dx = nn % 2

        dr = 2 * dx - 1
        xyz = nXYZ.copy()
        xyz[axis] += dr

        neighbor_lists.append(__xyzToOList(xyz, ownDepth))

    return neighbor_lists


@nb.njit()  # ,parallel=True)
def reverse_octant_list_to_xyz(octant_list):
    """
    Get xyz indices at octant origin from reversed parent list.

    Parameters
    ----------
    octant_list : int8[:]
        Indices of each parent octant within its parent, in reversed order.

    Returns
    -------
    XYZ : uint64[3]
        XYZ triple of smallest octant node.

    """
    depth = octant_list.shape[0]

    xyz = np.zeros(3, dtype=np.int64)
    for nn in nb.prange(depth):
        for jj in nb.prange(3):
            bit = (octant_list[nn] >> jj) & 1
            # xyz[jj]+=bit<<nn
            xyz[jj] += bit << (depth - nn - 1)

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


class Logger:
    """Monitor timing, memory use, and step progress."""

    def __init__(self, step_name, printout=False):
        """
        Start timing step execution.

        Parameters
        ----------
        step_name : string
            Name of the step.
        printout : bool, optional
            Whether to print progress to stdout. The default is False.

        Returns
        -------
        None.

        """
        self.name = step_name
        self.printout = printout
        if printout:
            print(step_name + " starting")
        self.startWall = time.monotonic()
        self.start = time.process_time()
        self.durationCPU = 0
        self.durationWall = 0
        self.memory = 0

    def logCompletion(self):
        """
        Log completion of step and prints duration if configured to.

        Returns
        -------
        None.

        """
        tWall = time.monotonic()
        tend = time.process_time()
        durationCPU = tend - self.start
        durationWall = tWall - self.startWall

        if self.printout:
            engFormat = tickr.EngFormatter()
            print(self.name + ": " + engFormat(durationCPU) + "s [CPU], " + engFormat(durationWall) + "s [wall]")
        self.durationCPU = durationCPU
        self.durationWall = durationWall
        # self.memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.memory = psutil.Process().memory_info().rss


def unravel_array_set(masked_arrays):
    """
    Get all unmasked values as a 1d array

    Parameters
    ----------
    masked_arrays : List of masked arrays
        Data to unravel

    Returns
    -------
    numeric[:]
        Nonmasked values of all arrays
    """
    vals = []
    for arr in masked_arrays:
        goodvals = arr.data[~arr.mask]
        vals.extend(goodvals.ravel())

    return np.array(vals)


def fastcount(boolArray):
    """
    Faster equivalent to summing boolean array

    Parameters
    ----------
    array : bool[:]
        Logical array.

    Returns
    -------
    int
        number of true elements.

    """
    return np.nonzero(boolArray)[0].shape[0]
