#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh topology
"""
import numpy as np
import numba as nb
from . import elements
from . import util
from . import fem
from . import geometry as geo
from tqdm import trange


class Mesh:
    def __init__(self, bbox, element_type="Admittance"):
        self.bbox = bbox
        self.extents = (bbox[3:] - bbox[:3]) / 2
        self.span = bbox[3:] - bbox[:3]
        self.center = bbox[:3] + self.span / 2
        self.elements = []
        self.conductances = []
        self.element_type = element_type
        self.node_coords = np.empty((0, 3), dtype=np.float64)
        self.edges = []

        self.minl0 = 0

        self.index_map = []
        self.inverse_index_map = {}
        self.boundary_nodes = np.empty(0, dtype=np.int64)

    def __getstate__(self):
        state = self.__dict__.copy()

        elInfo = []
        for el in self.elements:
            d = {
                "origin": el.origin,
                "span": el.span,
                "l0": el.l0,
                "sigma": el.sigma,
                "vertices": el.vertices,
                "faces": el.faces,
                "index": el.index,
            }
            elInfo.append(d)

        state["elements"] = elInfo
        state["inverse_index_map"] = {}
        if "tree" in state:
            dtree = {
                "origin": self.tree.origin,
                "span": self.tree.span,
                "sigma": self.tree.sigma,
                "index": self.tree.index,
            }
            state["tree"] = dtree
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        elDicts = self.elements.copy()

        self.elements = []

        for ii, el in enumerate(elDicts):
            self.add_element(el["origin"], el["span"], el["sigma"], el["index"])
            self.elements[-1].faces = el["faces"]
            self.elements[-1].vertices = el["vertices"]

        self.inverse_index_map = util.get_index_dict(self.index_map)

        if "tree" in state:
            tree_dict = self.tree.copy()
            tree = Octant(
                origin=tree_dict["origin"],
                span=tree_dict["span"],
                sigma=tree_dict["sigma"],
                index=tree_dict["index"],
            )
            tree._Octant__recreateTree(self.elements)
            self.tree = tree

    def get_containing_element(self, coords):
        """
        Get element containing specified point.

        Parameters
        ----------
        coords : float[:]
            Cartesian coordinates of point.

        Raises
        ------
        ValueError
            Error if no element contains the point.

        Returns
        -------
        elem : xcell.Element
            Containing element.

        """
        nElem = len(self.elements)

        # TODO: workaround fudge factor
        tol = 1e-9 * np.ones(3)

        for nn in nb.prange(nElem):
            elem = self.elements[nn]
            delta = coords - elem.origin + tol
            ext = elem.span + 2 * tol

            difs = np.logical_and(delta >= 0, delta <= ext)
            if all(difs):
                return elem

        raise ValueError("Point (%s) not inside any element" % ",".join(map(str, coords)))

    def finalize(self):
        pass

    def add_element(self, origin, span, sigma, index):
        """
        Insert element into the mesh.

        Parameters
        ----------
        origin : float[:]
            Cartesian coords of element's origin.
        extents : float[:]
            Length of edges in x,y,z.
        sigma : float
            Conductivity of element.
        nodeIndices : int64[:]
            Numbering of element nodes according to global mesh.

        Returns
        -------
        None.

        """

        newEl = Octant(origin, span, sigma=sigma, index=index)
        self.elements.append(newEl)

    def getConductances(self, elements=None):
        """
        Get the discrete conductances from every element.


        Parameters
        ----------
        elements : list of `~xcell.mesh.Octant`, optional
            DESCRIPTION. The default is None, which uses all in mesh.

        Returns
        -------
        edge_indices : int64[:,:]
            List of node pairs spanned by each conductance.
        conductances : float
            Conductance in siemens.
        transforms : list of ints
            Substitutions for graph-dual meshes.

        """

        if elements is None:
            elements = self.elements
        nElem = len(elements)

        clist = []
        elist = []
        transforms = []

        for nn in nb.prange(nElem):
            elem = elements[nn]

            if self.element_type == "Admittance":
                element_conductances = fem._get_admittance_conductances(elem.span, elem.sigma)
                element_edges = elem.vertices[fem.ADMITTANCE_EDGES]
            elif self.element_type == "FEM":
                element_conductances = fem.get_hex_conductances(elem.span, elem.sigma)
                element_edges = elem.vertices[fem.HEX_EDGES]
            elif self.element_type == "Face":
                raw_conductances = fem.get_face_conductances(elem.span, elem.sigma)
                raw_edges = elem.faces[fem.FACE_EDGES]

                element_conductances = []
                element_edges = []
                for ii in nb.prange(6):
                    neighbors = elem.neighbors[ii]
                    nNei = len(neighbors)

                    if nNei <= 1:
                        element_conductances.append(raw_conductances[ii])
                        element_edges.append(raw_edges[ii])

                    else:
                        # faces are shared; split original edge
                        origNode = raw_edges[ii, 0]
                        xform = []

                        neiNodeIdx = ii + (-1) ** ii

                        for jj in nb.prange(nNei):
                            element_conductances.append(raw_conductances[ii] / nNei)

                            neighbor = neighbors[jj]
                            neiNode = neighbor.faces[neiNodeIdx]

                            edge = np.array([neiNode, raw_edges[ii, 1]])
                            element_edges.append(edge)
                            xform.append(neiNode)

                        xform.append(origNode)
                        transforms.append(xform)

            elist.extend(element_edges)
            clist.extend(element_conductances)

        edge_indices = np.array(elist)
        conductances = np.array(clist)

        return edge_indices, conductances, transforms

    def get_min_l0(self):
        """
        Get the smallest edge length in mesh

        Returns
        -------
        min_l0 : float
            smallest edge length.

        """
        min_l0 = np.infty

        for el in self.elements:
            min_l0 = min(min_l0, el.l0)
        return min_l0

    def get_boundary_nodes(self):
        """
        Get the indices of nodes on the domain boundary.

        Returns
        -------
        global_indices : int[:]
            Indices of current mesh nodes at periphery.

        """
        mins, maxes = np.hsplit(self.bbox, 2)

        atmin = np.equal(mins, self.node_coords)
        atmax = np.equal(maxes, self.node_coords)
        isbnd = np.any(np.logical_or(atmin, atmax), axis=1)
        global_indices = np.nonzero(isbnd)[0]

        return global_indices

    def get_intersecting_elements(self, axis, coordinate):
        """
        Find elements intersected by a cartesian plane.

        Parameters
        ----------
        axis : int
            Normal of intersecting plane (0->x, 1->y,2->z).
        coordinate : float
            Coordinate of plane alonx normal axis.

        Returns
        -------
        elements : list of `~xcell.meshes.Octant`
            Elements intersected by the plane.

        """
        elements = []

        for el in self.elements:
            gt = el.origin[axis] >= coordinate
            lt = (el.origin[axis] - el.span[axis]) < coordinate
            if gt and lt:
                elements.append(el)

        return elements


class Octree(Mesh):
    def __init__(self, bounding_box, max_depth=10, element_type="Admittance"):
        self.center = np.mean(bounding_box.reshape(2, 3), axis=0)
        self.span = bounding_box[3:] - bounding_box[:3]
        super().__init__(bounding_box, element_type)

        self.max_depth = max_depth
        self.bbox = bounding_box
        self.index_map = np.empty(0, dtype=np.uint64)
        self.inverse_index_map = {}

        self.changed = True

        self.tree = Octant(origin=bounding_box[:3], span=self.span)

    def get_containing_element(self, coords):
        el = self.tree.get_containing_element(coords)
        return el

    def get_intersecting_elements(self, axis, coordinate):
        elements = self.tree.get_intersecting_elements(axis, coordinate)
        return elements

    def refine_by_metric(self, min_l0_function, ref_pts, max_l0_function=None, coefs=None, coarsen=True):
        """
        Recursively splits elements until l0_function evaluated at the center
        of each element is greater than that element's l0'

        Parameters
        ----------
        min_l0_function : function
            Function returning a scalar for each input cartesian coordinate.
        ref_pts : float[:,3]
            Cartesian coordinates where distance is evaluated from.
        max_l0_function : function or None, optional
            Function giving maximum l0 for coarsening.
            The default is None, which uses min_l0_function.
        coefs : float[:], optional
            Factor multiplied each candidate l0. The default is None.
        coarsen : bool, optional
            Whether to prune elements larger than target l0. The default is True.

        Returns
        -------
        changed : bool
            Adaptation resulted in new topology

        """

        if max_l0_function is None:
            max_l0_function = min_l0_function

        changed = self.tree.refine_by_metric(min_l0_function, ref_pts, self.max_depth, coefs)

        if coarsen:
            pruned, _ = self.tree.coarsen_by_metric(max_l0_function, ref_pts, self.max_depth, coefs)

            self.changed = changed | pruned

        return changed

    def finalize(self):
        """
        Convert terminal octants to mesh elements, mapping sparse to dense numbering.

        Returns
        -------
        None.

        """
        octs = self.tree.get_terminal_octants()

        self.elements = octs

        if self.element_type == "Face":
            self.get_element_adjacencies()

        self.changed = False

        return

    def print_structure(self):
        """
        Debug tool to print structure of tree

        Returns
        -------
        None.

        """
        self.tree.print_structure()

    def octantByList(self, indexList, octant=None):
        """
        Select an octant by recursing through a list of indices.

        Parameters
        ----------
        indexList : int64[]
            Octant identifier, where i=indexList[n] specifies child i of octant n
        octant : Octant, optional
            Used internally to recurse. The default is None.

        Returns
        -------
        Octant
            The octant object specified.

        """
        head = indexList.pop(0)

        if octant is None:
            octant = self.tree
        oc = octant.children[head]
        if len(oc.children) == 0:
            # terminal octant reached, match found
            return [oc]
        else:
            if len(indexList) == 0:
                # list exhausted at non-terminal octant
                return oc.children
            else:
                # continue recursion
                return self.octantByList(indexList, oc)

    def countelements(self):
        """
        Get number of terminal elements in tree.

        Returns
        -------
        int
            Number of leaf elements.

        """
        return self.tree.countelements()

    def get_coords_recursively(self):
        """
        Determine coordinates of mesh nodes.

        Recurses through `Octant.get_coords_recursively`

        Returns
        -------
        coords : 2d float array
            Cartesian coordinates of each mesh node.

        """

        # TODO: parallelize?
        # around 2x element creation time (not bad)
        # rest of function very fast

        asdual = self.element_type == "Face"
        i = self.tree.get_coords_recursively(asDual=asdual)

        indices = np.unique(np.array(i, dtype=np.uint64))
        self.index_map = indices

        # c=util.indices_to_coordinates(indices,self.span,self.max_depth+int(asdual))
        # coords=c+self.bbox[:3]
        coords = util.indices_to_coordinates(indices, self.bbox[:3], self.span)

        return coords

    def get_boundary_nodes(self):
        """
        Get the indices of nodes on the domain boundary.

        Returns
        -------
        global_indices : int[:]
            Indices of current mesh nodes at periphery.

        """
        bnodes = []

        for ii in nb.prange(self.index_map.shape[0]):
            nn = self.index_map[ii]
            xyz = util.index_to_xyz(nn, util.MAXPT)
            if np.any(xyz == 0) or np.any(xyz == util.MAXPT - 1):
                bnodes.append(ii)

        return np.array(bnodes)

    def get_element_adjacencies(self):
        """
        Get the neighboring elements.

        Returns
        -------
        adjacencies : list of elements
            Adjacent elements in order of (+x,-x, +y,-y, +z,-z).

        """
        numel = len(self.elements)

        adjacencies = []

        for ii in trange(numel, desc="Calculating adjacency"):
            el = self.elements[ii]

            maybeNeighbors = util.get_octant_neighbor_lists(np.array(el.index))
            neighborSet = []
            for nn in nb.prange(6):
                nei = maybeNeighbors[nn]
                if len(nei) > 0:
                    tstEl = self.octantByList(nei)

                    if len(tstEl) > 1:
                        ax = nn // 2
                        dr = nn % 2

                        # select elements on opposite side of
                        # sameFace=[((n>>ax)&1) for n in range(6)]
                        # neighbors=[tstEl[n] for n in sameFace]

                        neighbors = [n for n in tstEl if util.to_bit_array(n.index[-1])[ax] ^ dr]
                        # neighbors=[n for n in tstEl if util.to_bit_array(n.index[-1])[ax]]

                    else:
                        neighbors = tstEl

                else:
                    neighbors = []

                neighborSet.append(neighbors)

            el.neighbors = neighborSet
            adjacencies.append(neighborSet)

        return adjacencies


class Octant:
    def __init__(self, origin, span, depth=0, sigma=np.ones(3), index=[], oXYZ=np.zeros(3, dtype=np.int32)):
        self.origin = origin
        self.span = span
        self.center = origin + span / 2
        self.bbox = np.concatenate((origin, origin + span))
        self.l0 = np.prod(span) ** (1 / 3)

        self.children = []
        self.depth = depth
        self.index = index

        # don't calculate indices here, since we might split later
        self.vertices = []
        self.faces = []

        self.sigma = sigma

        self.neighbors = 6 * [[]]
        self.oXYZ = oXYZ

    def __recreateTree(self, elements):
        childLists = []
        D = self.depth
        for ind in range(8):
            childList = [el for el in elements if el.index[D] == ind]
            childLists.append(childList)

        self.split()
        for ii, ch in enumerate(self.children):
            clist = childLists[ii]
            if len(clist) == 1:
                el = clist[0]
                el.depth = len(el.index)
                self.children[ii] = el
            else:
                ch.__recreateTree(clist)

    def countelements(self):
        """
        Return the number of leaf elements contained within octant.

        Returns
        -------
        int
            Number of leaf elements within.

        """
        if len(self.children) == 0:
            return 1
        else:
            return sum([ch.countelements() for ch in self.children])

    # @nb.njit(parallel=True)
    def split(self, division=np.array([0.5, 0.5, 0.5])):
        """
        Split element into its child octants.

        Parameters
        ----------
        division : float[3], optional
            Fraction of division in x,y,z directions. The default is np.array([0.5, 0.5, 0.5]).

        Returns
        -------
        None.

        """
        newSpan = self.span * division

        for ii in nb.prange(8):
            offset = newSpan * util.OCT_INDEX_BITS[ii]
            # offset=util.to_bit_array(ii)*newSpan
            newOrigin = self.origin + offset
            newIndex = self.index.copy()
            newIndex.append(ii)

            self.children.append(Octant(origin=newOrigin, span=newSpan, depth=self.depth + 1, index=newIndex))

    def get_coords_recursively(self, asDual=False):
        """
        Get the coordinates of the mesh within the element.

        Parameters
        ----------
        asDual : bool, optional
            Whether to return the mesh-dual nodes instead of vertices. The default is False.

        Returns
        -------
        indices : list of int
            Indices of node according to universal numbering.

        """
        if len(self.children) == 0:
            if asDual:
                indices = self.faces.tolist()
            else:
                indices = self.vertices.tolist()
        else:
            indices = []

            for ch in self.children:
                i = ch.get_coords_recursively(asDual=asDual)
                indices.extend(i)

        return indices

    # @nb.njit(parallel=True)
    def refine_by_metric(self, l0_function, ref_pts, max_depth, coefs):
        """
        Recursively splits elements until l0_function evaluated at the center
        of each element is greater than that element's l0'

        Parameters
        ----------
        min_l0_function : function
            Function returning a scalar for each input cartesian coordinate.
        ref_pts : float[:,3]
            Cartesian coordinates where distance is evaluated from.
        max_depth : int
            Maximum depth of splitting permitted
        coefs : float[:], optional
            Factor multiplied each candidate l0. The default is None.

        Returns
        -------
        changed : bool
            Adaptation resulted in new topology

        """
        changed = False
        which_pts = util.reduce_functions(l0_function, ref_pts, self.bbox, coefs=coefs)
        filt = np.logical_and(which_pts, max_depth > self.depth)
        nextPts = ref_pts[filt]
        nextmax_depths = max_depth[filt]

        if coefs is not None:
            next_coefs = coefs[filt]
        else:
            next_coefs = coefs

        if nextPts.shape[0] > 0:  # and self.depth<max_depth:
            if len(self.children) == 0:
                changed = True
                self.split()
            for ii in nb.prange(8):
                changed |= self.children[ii].refine_by_metric(l0_function, nextPts, nextmax_depths, next_coefs)

        return changed

    def coarsen_by_metric(self, metric, ref_pts, max_depth, coefs):
        """
        Delete children if element and all children are smaller than target.

        Parameters
        ----------
        metric : function
            Function returning a scalar for each input cartesian coordinate.
        ref_pts : float[:,3]
            Cartesian coordinates where distance is evaluated from.
        max_depth : int
            Maximum depth of splitting.
        coefs : float[:]
            Factor multiplied each candidate l0.

        Returns
        -------
        changed : bool
            Whether mesh topology was altered.
        undersized : bool
            Whether element is smaller than all targets.

        """
        changed = False
        which_pts = util.reduce_functions(metric, ref_pts, self.bbox, coefs=coefs, return_under=False)

        # let metric implicitly prune if max_depth lowered
        # causes insufficient meshing otherwise
        # undersize=np.all(which_pts) #or self.depth<max_depth

        filt = np.logical_or(which_pts, max_depth < self.depth)
        undersize = np.all(filt)

        if self.is_terminal():
            # end condition
            pass
        else:
            # recurse to end
            for ch in self.children:
                child_changed, child_under = ch.coarsen_by_metric(metric, ref_pts, max_depth, coefs)
                changed |= child_changed
                undersize &= child_under

            if undersize:
                changed = True
                for ch in self.children:
                    del ch
                self.children = []
                self._calculate_indices()

        return changed, undersize

    def print_structure(self):
        """
        Print out octree structure.

        Returns
        -------
        None.

        """
        base = "> " * self.depth
        print(base + str(self.l0))

        for ch in self.children:
            ch.print_structure()

    def is_terminal(self):
        """
        Determine if element is terminal (has no children)

        Returns
        -------
        terminal : bool
            True if element has no children.

        """
        terminal = len(self.children) == 0
        if terminal and len(self.vertices) == 0:
            self._calculate_indices()
        return terminal

    def intersects_plane(self, normal, coord):
        """
        Calculate whether plane intersects element

        Parameters
        ----------
        normal : int or bool[3]
            Which axis (x,y,z) contains the plane normal.
        coord : float
            Coordinate of plane along its normal.

        Returns
        -------
        bool
            Whether plane intersects element
        """
        cmin = self.origin[normal]
        cmax = cmin + self.span[normal]
        return (cmin <= coord) & (coord < cmax)

    def get_terminal_octants(self):
        """
        Get all childless octants.

        Returns
        -------
        list of Octants
            Childless octants (the actual elements of the mesh)

        """
        if self.is_terminal():
            return [self]
        else:
            descendants = []
            for ch in self.children:
                # if len(ch.children)==0:
                grandkids = ch.get_terminal_octants()

                if grandkids is not None:
                    descendants.extend(grandkids)
        return descendants

    def _calculate_indices(self):
        """
        Calculate the universal indices of the element's nodes.

        Returns
        -------
        None.

        """
        parent_list = np.array(self.index, dtype=np.int8)
        inds = util.get_indices_of_octant(parent_list, fem.HEX_POINT_INDICES)

        self.vertices = inds[:8]
        self.faces = inds[8:]

    def get_containing_element(self, coords):
        """
        Find the element that contains the specified point.

        Parameters
        ----------
        coords : float[3]
            Cartesian coordinates of test point.

        Returns
        -------
        xcell.meshes.Octant or None
            Element containing point.

        """
        if len(self.children) == 0:
            # if self.containsPoint(coords):
            if geo.is_in_bbox(self.bbox, coords):
                return self
            else:
                return None
        else:
            for ch in self.children:
                tmp = ch.get_containing_element(coords)
                if tmp is not None:
                    return tmp
            return None

    def get_intersecting_elements(self, axis, coord):
        """
        Find elements intersected by a cartesian plane.

        Parameters
        ----------
        axis : int
            Normal of intersecting plane (0->x, 1->y,2->z).
        coord : float
            Coordinate of plane alonx normal axis.

        Returns
        -------
        elements : list of `~xcell.meshes.Octant`
            Elements intersected by the plane.

        """
        descendants = []
        if len(self.children) == 0:
            if self.intersects_plane(axis, coord):
                return [self]
            else:
                return descendants
        else:
            for ch in self.children:
                intersects = ch.get_intersecting_elements(axis, coord)
                if intersects is not None:
                    descendants.extend(intersects)

            return descendants

    def interpolate_within(self, coordinates, values):
        """
        Interpolate values within the element from specified vertices or face nodes

        Parameters
        ----------
        coordinates : float[:,3]
            Cartesian coordinates to interpolate at (in global coordinate system).
        values : float[8] or float[7]
            Array of values at vertices (n=8) or faces (n=7).

        Returns
        -------
        interp : float[:]
            Interpolated values at the specified points.

        """
        coords = fem.to_local_coords(coordinates, self.center, self.span)
        if values.shape[0] == 8:
            interp = fem.interpolate_from_verts(values, coords)
        else:
            interp = fem.interpolate_from_face(values, coords)

        return interp

    # TODO: fix hack of changing node indices
    def get_universal_vals(self, known_values):
        """
        Get the values at all face and vertex nodes of the element,
        interpolating from the supplied face or vertex values.

        Parameters
        ----------
        known_values : float[7] or float[8]
            Values at face (n=7) or vertices (n=8) of element

        Returns
        -------
        all_values : float[15]
            Values at all universal points
        all_indices : uint64[15]
            Global indices of each point
        """
        all_values = np.empty(15, dtype=np.float64)
        all_indices = np.empty(15, dtype=np.int64)

        vertex_indices = self.vertices
        face_indices = self.faces

        if known_values.shape[0] == 8:
            vertex_values = known_values
            face_values = fem.interpolate_from_verts(known_values, fem.HEX_FACE_COORDS)
        else:
            face_values = known_values
            vertex_values = fem.interpolate_from_face(known_values, fem.HEX_VERTEX_COORDS)

        all_indices = np.concatenate((vertex_indices, face_indices))
        all_values = np.concatenate((vertex_values, face_values))

        return all_values, all_indices

    def get_planar_values(self, global_values, axis=2, coord=0.0):
        """
        Interpolate values where a plane intersects the element.

        Parameters
        ----------
        global_values : float[7] or float[8]
            Values at faces (n=7) or vertices (n=8) of element
        axis : int or bool[3], optional
            Which of (x,y,z) is the plane's normal. Defaults to 2 (z)
        coord : float, optional
            Coordinate of plane along its normal. Defaults to 0.0

        Returns
        -------
        float[4]
            Interpolated values at corners of plane-element intersection.
        """
        zcoord = (coord - self.center[axis]) / self.span[axis]

        inds = 3 * [[-1.0, 1.0]]
        inds[axis] = [zcoord]
        local_coords = np.array([[x, y, z] for z in inds[2] for y in inds[1] for x in inds[0]])

        if global_values.shape[0] == 8:
            plane_values = fem.interpolate_from_verts(global_values, local_coords)
        else:
            plane_values = fem.interpolate_from_face(global_values, local_coords)

        return plane_values
