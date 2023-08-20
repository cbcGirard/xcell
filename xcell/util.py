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
MAXPT = np.array(2**(MAXDEPTH+1)+1, dtype=np.uint64)[()]

def point_current_source_voltage(eval_coords, i_source, sigma=1., source_location=np.zeros(3, dtype=np.float64)):
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
    dif = eval_coords-source_location
    k = i_source/(4*np.pi*sigma)
    v = np.array([k/np.linalg.norm(d) for d in dif])
    return v


def disk_current_source_voltage(eval_coords, i_source, sigma=1., source_location=np.zeros(3, dtype=np.float64)):
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
    dif = eval_coords-source_location
    k = i_source/(4*sigma)
    v = np.array([k/np.linalg.norm(d) for d in dif])
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
    factor = 10**np.floor(np.log10(x))

    return factor*np.ceil(x/factor)


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
    return 10**np.floor(np.log10(val))


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
    return 10**np.ceil(np.log10(val))


def loground(axis, which='both'):
    """
    Set plot axes' bounds to powers of 10.

    Parameters
    ----------
    axis : `matplotlib.axes.Axes`
        Plot axes to alter.
    which : str, optional
        Which axes to change bounds: 'x', 'y', or 'both' (default).
    """
    lims = [[logfloor(aa[0]), logceil(aa[1])]
            for aa in axis.dataLim.get_points().transpose()]  # [xl, yl]]
    if which == 'x' or which == 'both':
        axis.set_xlim(lims[0])
    if which == 'y' or which == 'both':
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
        xx = 0.5*(xx[:-1]+xx[1:])
        yy = 0.5*(yy[:-1]+yy[1:])

    XX, YY = np.meshgrid(xx, yy)

    pts = np.vstack((XX.ravel(), YY.ravel())).transpose()
    return pts



# TODO: remove unused(?) class
# Attempt to create faster, Numba implementation of map
# from integer tag to position in list/array of integers
# e.g. map from universal node index to mesh node index
# Need further testing to see if speed benefit is possible
#
# @nb.experimental.jitclass()
# class index_map:
#     def __init__(self, sparse_indices):
#         self.sparse = sparse_indices
#
#     def _find_match(self, value, lower, upper):
#         if lower == upper:
#             match = lower
#         else:
#             mid = (lower+upper)//2
#             if self.sparse[mid] > value:
#                 match = self._find_match(value, lower, mid)
#             else:
#                 match = self._find_match(value, mid, upper)
#
#         return match

#     def to_dense(self, sparse_index):
#         match = self._findMatch(sparse_index, 0, self.sparse.shape[0])
#         return match


# TODO: remove unused function
# def eliminate_rows(matrix):
#     N = matrix.shape[0]
#     # lo=scipy.sparse.tril(matrix,-1)
#     lo = matrix.copy()
#     lo.sum_duplicates()
#
#     rowIdx, colIdx = lo.nonzero()
#     _, count = np.unique(rowIdx, return_counts=True)
#
#     tup = __eliminate_rowsLoop(rowIdx, colIdx, count)
#     v, r, c = tup
#
#     # xmat=scipy.sparse.coo_matrix(tup,
#     #                              shape=(N,N))
#     xmat = scipy.sparse.coo_matrix((v, (r, c)),
#                                    shape=(N, N))
#
#     return xmat

# TODO: remove unused function
# def htile(array, ncopy):
#     return np.vstack([array]*ncopy).transpose()

# TODO: remove unused function
# @nb.njit(nb.types.Tuple(nb.float64[:], nb.int64[:], nb.int64[:])(nb.int64[:], nb.int64[:], nb.int64[:]),
#          parallel=True)
# @nb.njit(parallel=True)
# def __eliminate_rowsLoop(rowIdx, colIdx, count):
#     r = []
#     c = []
#     v = []
#     hanging = count < 6
#     for ii in nb.prange(count.shape[0]):
#         if hanging[ii]:
#             nconn = count[ii]-1
#             cols = colIdx[rowIdx == ii]
#             cols = cols[cols != ii]

#             vals = np.ones(nconn)/nconn

#             if vals.shape != cols.shape:
#                 print()

#             for jj in nb.prange(cols.shape[0]):
#                 c.append(cols[jj])
#                 r.append(ii)
#                 v.append(vals[jj])

#         else:
#             r.append(ii)
#             c.append(ii)
#             v.append(1.0)

#     # tupIdx=(np.array(r),np.array(c))
#     # tup=(np.array(v),tupIdx)
#     # tup=(np.array(v), np.array(r), np.array(c))
#     return (v, r, c)
#     # return tup

# TODO: remove unused function?
# def deduplicate_edges(edges, conductances):
#     """
#     Combine conductances that are in parallel.
#
#     Parameters
#     ----------
#     edges : int[:,2]
#         Global indices of the edge endpoints.
#     conductances : float[:]
#         Discrete conductance between the nodes.
#
#     Returns
#     -------
#     new_edges : int[:,2]
#         Minimal set of conductances' endpoints.
#     new_conductances : float[:]
#         Minimal set of conductances.
#
#     """
#     N = max(edges.ravel())+1
#     mat = scipy.sparse.coo_matrix((conductances,
#                                    (edges[:, 0], edges[:, 1])),
#                                   shape=(N, N))
#
#     dmat = mat.copy()+mat.transpose()
#
#     dedup = scipy.sparse.tril(dmat, k=-1)
#
#     new_edges = np.array(dedup.nonzero()).transpose()
#
#     new_conductances = dedup.data
#
#     return new_edges, new_conductances

# TODO: test alternative to get_index_dict
# @nb.njit
# def condense_indices(global_mask):
#     """
#     Get array for mapping local to global numbering.
#
#    .. deprecated
#        Lower mem usage, but horribly slow.
#        Use `.get_index_dict` instead
#
#     Parameters
#     ----------
#     global_mask : bool array
#         Global elements included in subset.
#
#     Returns
#     -------
#     whereSubset : int array
#         DESCRIPTION.
#
#     """
#     # whereSubset=-np.ones_like(global_mask)
#     whereSubset = np.empty_like(global_mask)
#     nSubset = fastcount(global_mask)
#     whereSubset[global_mask] = np.arange(nSubset)
#
#     return whereSubset



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

# TODO: marked for deletion
# @nb.njit(parallel=True)
# def octreeLoop_get_boundary_nodesLoop(nX, index_map):
#     """
#     Numba loop to return the boundary nodes of a mesh.

#     Parameters
#     ----------
#     nX : TYPE
#         DESCRIPTION.
#     index_map : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     bnodes = []
#     for ii in nb.prange(index_map.shape[0]):
#         nn = index_map[ii]
#         xyz = index_to_xyz(nn, nX)
#         if np.any(xyz == 0) or np.any(xyz == (nX-1)):
#             bnodes.append(ii)

#     return np.array(bnodes, dtype=np.int64)


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

    raise ValueError('not a member of dense_list')


# TODO: marked for deletion
# def coords2masked_arrays(intCoords, edges, planeMask, vals):
#     pcoords = np.ma.masked_array(intCoords, mask=~planeMask.repeat(2))
#     edgeInPlane = np.all(planeMask[edges], axis=1)
#     pEdges = np.ma.masked_array(edges, mask=~edgeInPlane.repeat(2))
#     edgeLs = abs(np.diff(np.diff(pcoords[pEdges], axis=2), axis=1)).squeeze()

#     span = pcoords.max()
#     edgeSizes = get_unmasked(np.unique(edgeLs))

#     arrays = []
#     for s in edgeSizes:
#         nArray = span//s+1
#         arr = np.nan*np.empty((nArray, nArray))

#         # get
#         edgesThisSize = np.ma.array(pEdges, mask=(edgeLs != s).repeat(2))
#         nodesThisSize, nConn = np.unique(
#             edgesThisSize.compressed(), return_counts=True)

#         # nodesThisSize[nConn<2]=np.ma.masked
#         # whichNodes=get_unmasked(nodesThisSize)
#         whichNodes = nodesThisSize

#         arrCoords = get_unmasked(pcoords[whichNodes])//s
#         arrI, arrJ = np.hsplit(arrCoords, 2)

#         arr[arrI.squeeze(), arrJ.squeeze()] = vals[whichNodes]
#         arrays.append(np.ma.masked_invalid(arr))

#     return arrays

# TODO: remove unused function
# def get_unmasked(mask_array):
#     if len(mask_array.shape) > 1:
#         isValid = np.all(~mask_array.mask, axis=1)
#     else:
#         isValid = ~mask_array.mask
#     return mask_array.data[isValid]

# TODO: marked for deletion
# def quadsTomasked_arrays(quadInds, quadVals):
#     arrays = []
#     quadSize = quadInds[:, 1]-quadInds[:, 0]
#     nmax = max(quadInds.ravel())

#     sizes = np.unique(quadSize)

#     for s in sizes:
#         which = quadSize == s
#         nGrid = nmax//s+1
#         vArr = np.nan*np.empty((nGrid, nGrid))
#         grid0s = quadInds[:, [0, 2]][which]//s
#         vgrids = quadVals[which]

#         # if nGrid==9:
#         #     print()

#         for ii in range(vgrids.shape[0]):
#             a, b = grid0s[ii]
#             v = vgrids[ii]
#             vArr[a, b] = v[0]
#             vArr[a+1, b] = v[1]
#             vArr[a, b+1] = v[2]
#             vArr[a+1, b+1] = v[3]

#         vmask = np.ma.masked_invalid(vArr)
#         arrays.append(vmask)
#     return arrays

# TODO: marked for deletion?
# @nb.njit(parallel=True)
# def edgeCurrentLoop(gList, edgeMat, dof2Global, vvec, gCoords, source_coords):
#     currents = np.empty_like(gList, dtype=np.float64)
#     nEdges = gList.shape[0]
#     edges = np.empty((nEdges, 2, 3),
#                      dtype=np.float64)
#     for ii in nb.prange(nEdges):
#         g = gList[ii]
#         dofEdge = edgeMat[ii]
#         globalEdge = dof2Global[dofEdge]

#         vs = vvec[dofEdge]
#         dv = vs[1]-vs[0]

#         if dv < 0:
#             dv = -dv
#             globalEdge = np.array([globalEdge[1], globalEdge[0]])

#         for pp in np.arange(2):
#             p = globalEdge[pp]
#             if p < 0:
#                 c = source_coords[-1-p]
#             else:
#                 c = gCoords[p]

#             edges[ii, pp] = c

#         # i=g*dv

#         currents[ii] = g*dv
#         # edges[ii]=np.array(coords)

#     return (currents, edges)

# TODO: marked for deletion
# @nb.njit()
# def edgeRoles(edges, node_role_table):
#     edgeRoles = np.empty_like(edges)
#     for ii in nb.prange(edges.shape[0]):
#         edgeRoles[ii] = node_role_table[edges[ii]]

#     return edgeRoles


# TODO: marked for deletion
# @nb.njit()
# def edgeNodesOfType(edges, nodeSelect):
#     N = edges.shape[0]
#     matches = np.empty(N, dtype=np.int64)

#     for ii in nb.prange(N):
#         e = edges[ii]
#         matches[ii] = np.sum(e)

#     return matches

# TODO: marked for deletion
# def getquads(x, y, xInt, yInt, values):
#     # x,y=np.hsplit(xy,2)
#     # x=xy[:,0]
#     # y=xy[:,1]
#     _, kx, nx = np.unique(xInt, return_index=True, return_inverse=True)
#     _, ky, ny = np.unique(yInt, return_index=True, return_inverse=True)

#     quadVals, quadCoords = __getquadLoop(x, y, kx, ky, nx, ny, values)

#     return quadVals, quadCoords

# TODO: marked for deletion
# @nb.njit(parallel=True)
# def __getquadLoop(x, y, kx, ky, nx, ny, values):
#     quadVals = []
#     quadCoords = []

#     sel = np.empty((4, x.shape[0]), dtype=np.bool8)

#     for yy in nb.prange(len(ky)-1):
#         for xx in nb.prange(len(kx)-1):

#             indices = __getSel(nx, ny, xx, yy)

#             if indices.shape[0] > 0:
#                 # x0,y0=xy[sel[0]][0]
#                 # x1,y1=xy[sel[3]][0]
#                 x0 = x[indices[0]]
#                 y0 = y[indices[0]]
#                 x1 = x[indices[3]]
#                 y1 = y[indices[3]]
#                 qcoords = np.array([x0, x1, y0, y1])
#                 # where=np.array([np.nonzero(s)[0][0] for s in sel])
#                 qvals = values[indices]
#                 # qvals=np.array([values[sel[n,:]] for n in np.arange(4)]).squeeze()

#                 quadVals.append(qvals)
#                 quadCoords.append(qcoords)

#     return np.array(quadVals), np.array(quadCoords)

# TODO: Remove unused
# @nb.njit
# def toBilinearVars(coord):
#     return np.array([1.,
#                      coord[0],
#                      coord[1],
#                      coord[0]*coord[1]])

# TODO: Remove unused
# @nb.njit
# def interpolateBilin(node_values, location):
#     locCoef = toBilinearVars(location)
#
#     interpCoefs = getBilinCoefs(node_values)
#
#     interpVal = np.dot(interpCoefs, locCoef)
#     return interpVal

# TODO: Remove unused
# @nb.njit()
# def getBilinCoefs(vals):
#     inv = np.array([[1.,  0.,  0.,  0.],
#                     [-1.,  1.,  0.,  0.],
#                     [-1.,  0.,  1.,  0.],
#                     [1., -1., -1.,  1.]])
#
#     interpCoefs = inv @ vals
#     return interpCoefs


@nb.njit
def minOver(target, vals):
    sel = vals >= target
    return min(vals[sel])


@nb.njit
def maxUnder(target, vals):
    sel = vals <= target
    return max(vals[sel])

# TODO: remove unused
# # @nb.njit
# # def getElementInterpolant(element,node_values):
# @nb.njit()
# def getElementInterpolant(node_values):
#     # coords=element.getOwnCoords()
#     # xx=np.arange(2,dtype=np.float64)
#     # # coords=np.array([[x,y,z] for z in xx for y in xx for x in xx])
#     # coords=np.array([[0.,0.,0.],
#     #                  [1.,0.,0.],
#     #                  [0.,1.,0.],
#     #                  [1.,1.,0.],
#     #                  [0.,0.,1.],
#     #                  [1.,0.,1.],
#     #                  [0.,1.,1.],
#     #                  [1.,1.,1.]])
#     # coefs=np.empty((8,8),dtype=np.float64)
#     # for ii in range(8):
#     #     coefs[ii]=coord2InterpVals(coords[ii])

#     # interpCoefs=np.linalg.solve(coefs, node_values)

#     im = np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#                    [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#                    [-1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
#                    [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#                    [1., -1., -1.,  1.,  0.,  0.,  0.,  0.],
#                    [1., -1.,  0.,  0., -1.,  1.,  0.,  0.],
#                    [1.,  0., -1.,  0., -1.,  0.,  1.,  0.],
#                    [-1.,  1.,  1., -1.,  1., -1., -1.,  1.]])
#     # interpCoefs=np.matmul(im,node_values)
#     interpCoefs = im @ node_values

#     return interpCoefs

# TODO: Remove unused
# @nb.njit
# def evalulateInterpolant(interp, location):

#     # coeffs of a, bx, cy, dz, exy, fxz, gyz, hxyz
#     varList = coord2InterpVals(location)

#     # interpVal=np.matmul(interp,varList)
#     interpVal = np.dot(interp, varList)

#     return interpVal

# TODO: remove unused
# @nb.njit
# def coord2InterpVals(coord):
#     x, y, z = coord
#     return np.array([1,
#                      x,
#                      y,
#                      z,
#                      x*y,
#                      x*z,
#                      y*z,
#                      x*y*z]).transpose()




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
        val += arr[ii]*2**(nbit-ii-1)
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

    actual_l0 = np.prod(element_bbox[3:]-element_bbox[:3])**(1/3)

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
        dot[ii] = np.sum(a[ii]*b)

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

# TODO: remove unused function
# @nb.njit(parallel=True)
# def bulkCalcOctantIndices(parent_lists,elDepths):
#     numel=parent_lists.shape[0]
#     inds=np.empty((numel,15),dtype=np.uint64)
#     for nel in nb.prange(numel):
#         parent_list=parent_lists[nel,:elDepths[nel]]
#         inds[nel,:]=get_indices_of_octant(parent_list,
#                                        HEX_POINT_INDICES)

#     return inds

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
    depth = MAXDEPTH-parent_list.shape[0]
    scale = 1 << depth

    absPos = np.empty_like(relative_position, dtype=np.uint64)
    for ii in nb.prange(npt):
        for jj in nb.prange(3):
            absPos[ii, jj] = origin[jj] + scale*relative_position[ii, jj]

    return absPos


@nb.njit(parallel=True,)
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
        coords[ii] = span*ijk/(MAXPT-1)+origin

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
        scale = 1 << (MAXDEPTH-ii)
        ind = octant_list[ii]
        for ax in nb.prange(3):
            if (ind >> ax) & 1:
                xyz[ax] += scale

    return xyz

def get_octant_neighbor_lists(own_index_list):
    '''
    Calculate the lists of indices for each neighbor of an octant.

    Parameters
    ----------
    own_index_list : int[:]
        List of parent octants within their parents.

    Returns
    -------
    neighbor_lists : list of int
        List of parent indices for the 6 neighboring octants.

    '''
    ownDepth = own_index_list.shape[0]
    nXYZ = reverse_octant_list_to_xyz(own_index_list)
    neighbor_lists = []
    # keep Numba type inference happy

    for nn in nb.prange(6):
        axis = nn//2
        dx = nn % 2

        dr = 2*dx-1
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
            print(step_name+" starting")
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
