# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as cltypes

Node = np.dtype([
    ('center_x', cltypes.float),
    ('center_y', cltypes.float),
    ('center_z', cltypes.float),
    ('edge_len', cltypes.float),
    ('is_leaf', cltypes.int),
    ('rsa', cltypes.int),
    ('lsa', cltypes.int),
    ('lia', cltypes.int),
    ('ria', cltypes.int),
    ('rsp', cltypes.int),
    ('lsp', cltypes.int),
    ('lip', cltypes.int),
    ('rip', cltypes.int),
    ('num_points', cltypes.int),
    ('first_point_offset', cltypes.int),
    ('depth', cltypes.int)
    ])


class OcTree:
    """
    Cells are subdivided into eight quadrants.

    A quadrant goes from [center - edge_len / 2, center + edge_len / 2).
    """
    def __init__(self, center, edge_len, capacity):
        self.center = center
        self.edge_len = edge_len

        # anterior nodes (z-positive)
        self.rsa = None
        self.rsa_center =\
            self.center[0] + self.edge_len / 4,\
            self.center[1] + self.edge_len / 4,\
            self.center[2] + self.edge_len / 4
        self.lsa = None
        self.lsa_center =\
            self.center[0] - self.edge_len / 4,\
            self.center[1] + self.edge_len / 4,\
            self.center[2] + self.edge_len / 4
        self.lia = None
        self.lia_center =\
            self.center[0] - self.edge_len / 4,\
            self.center[1] - self.edge_len / 4,\
            self.center[2] + self.edge_len / 4
        self.ria = None
        self.ria_center =\
            self.center[0] + self.edge_len / 4,\
            self.center[1] - self.edge_len / 4,\
            self.center[2] + self.edge_len / 4

        # posterior nodes (z-negative)
        self.rsp = None
        self.rsp_center =\
            self.center[0] + self.edge_len / 4,\
            self.center[1] + self.edge_len / 4,\
            self.center[2] - self.edge_len / 4
        self.lsp = None
        self.lsp_center =\
            self.center[0] - self.edge_len / 4,\
            self.center[1] + self.edge_len / 4,\
            self.center[2] - self.edge_len / 4
        self.lip = None
        self.lip_center =\
            self.center[0] - self.edge_len / 4,\
            self.center[1] - self.edge_len / 4,\
            self.center[2] - self.edge_len / 4
        self.rip = None
        self.rip_center =\
            self.center[0] + self.edge_len / 4,\
            self.center[1] - self.edge_len / 4,\
            self.center[2] - self.edge_len / 4

        self.capacity = capacity
        self.points = []

    def shortest_distance_to_edges(self, point, abounds, bbounds):
        a, b = point
        v0 = min(np.sqrt((a - abounds[0])**2 + (b - bbounds[0])**2),
                 np.sqrt((a - abounds[0])**2 + (b - bbounds[1])**2))
        v1 = min(np.sqrt((a - abounds[1])**2 + (b - bbounds[0])**2),
                 np.sqrt((a - abounds[1])**2 + (b - bbounds[1])**2))
        return min(v0, v1)

    def shortest_distance_to_octant(self, point, center):
        """
        Return the shortest distance from the point to the quadrant
        centered on center.
        """
        x, y, z = point
        xbounds = center[0] - self.edge_len / 4, center[0] + self.edge_len / 4
        ybounds = center[1] - self.edge_len / 4, center[1] + self.edge_len / 4
        zbounds = center[2] - self.edge_len / 4, center[2] + self.edge_len / 4
        if xbounds[0] <= x < xbounds[1] and\
            ybounds[0] <= y < ybounds[1] and\
                zbounds[0] <= z < zbounds[1]:
            # point is inside the quadrant
            return -1.0
        elif xbounds[0] <= x < xbounds[1] and ybounds[0] <= y < ybounds[1]:
            return min(np.abs(z - zbounds[0]), np.abs(z - zbounds[1]))
        elif ybounds[0] <= y < ybounds[1] and zbounds[0] <= z < zbounds[1]:
            return min(np.abs(x - xbounds[0]), np.abs(x - xbounds[1]))
        elif zbounds[0] <= z < zbounds[1] and xbounds[0] <= x < xbounds[1]:
            return min(np.abs(y - ybounds[0]), np.abs(y - ybounds[1]))
        elif xbounds[0] <= x < xbounds[1]:
            return self.shortest_distance_to_edges((y, z), ybounds, zbounds)
        elif ybounds[0] <= y < ybounds[1]:
            return self.shortest_distance_to_edges((x, z), xbounds, zbounds)
        elif zbounds[0] <= z < zbounds[1]:
            return self.shortest_distance_to_edges((x, y), xbounds, ybounds)
        else:
            # point isn't in any of the quadrant ranges
            return min(
                np.sqrt((x - xbounds[0])**2 + (y - ybounds[0])**2
                        + (z - zbounds[0])**2),
                np.sqrt((x - xbounds[0])**2 + (y - ybounds[0])**2
                        + (z - zbounds[1])**2),
                np.sqrt((x - xbounds[1])**2 + (y - ybounds[0])**2
                        + (z - zbounds[0])**2),
                np.sqrt((x - xbounds[1])**2 + (y - ybounds[0])**2
                        + (z - zbounds[1])**2),
                np.sqrt((x - xbounds[0])**2 + (y - ybounds[1])**2
                        + (z - zbounds[0])**2),
                np.sqrt((x - xbounds[0])**2 + (y - ybounds[1])**2
                        + (z - zbounds[1])**2),
                np.sqrt((x - xbounds[1])**2 + (y - ybounds[1])**2
                        + (z - zbounds[0])**2),
                np.sqrt((x - xbounds[1])**2 + (y - ybounds[1])**2
                        + (z - zbounds[1])**2))

    def is_in_octant(self, point, center):
        """
        Return True if the point is contained in the quadrant.
        """
        return self.shortest_distance_to_octant(point, center) < 0.0

    def is_leaf(self):
        """
        Return True if the cell is a leaf.
        """
        return self.rsa is None and self.lsa is None and\
            self.lia is None and self.ria is None and\
            self.rsp is None and self.lsp is None and\
            self.lip is None and self.rip is None

    def insert(self, points):
        """
        Insert list of points into the octree.
        """
        # add points to the current node
        self.points.extend(points)

        # if the node is a leaf, test if it needs to be subdivided
        if self.is_leaf():
            if len(self.points) <= self.capacity:
                # no need to subdivide cell
                return

            # else subdivide the node
            self.rsa = OcTree(self.rsa_center, self.edge_len/2, self.capacity)
            self.lsa = OcTree(self.lsa_center, self.edge_len/2, self.capacity)
            self.lia = OcTree(self.lia_center, self.edge_len/2, self.capacity)
            self.ria = OcTree(self.ria_center, self.edge_len/2, self.capacity)
            self.rsp = OcTree(self.rsp_center, self.edge_len/2, self.capacity)
            self.lsp = OcTree(self.lsp_center, self.edge_len/2, self.capacity)
            self.lip = OcTree(self.lip_center, self.edge_len/2, self.capacity)
            self.rip = OcTree(self.rip_center, self.edge_len/2, self.capacity)

        # if it is not a leaf, recursively insert the points into child nodes
        rsa_points = []
        lsa_points = []
        lia_points = []
        ria_points = []
        rsp_points = []
        lsp_points = []
        lip_points = []
        rip_points = []
        for p in self.points:
            if self.is_in_octant(p, self.rsa_center):
                rsa_points.append(p)
            elif self.is_in_octant(p, self.lsa_center):
                lsa_points.append(p)
            elif self.is_in_octant(p, self.lia_center):
                lia_points.append(p)
            elif self.is_in_octant(p, self.ria_center):
                ria_points.append(p)
            elif self.is_in_octant(p, self.rsp_center):
                rsp_points.append(p)
            elif self.is_in_octant(p, self.lsp_center):
                lsp_points.append(p)
            elif self.is_in_octant(p, self.lip_center):
                lip_points.append(p)
            elif self.is_in_octant(p, self.rip_center):
                rip_points.append(p)

        if len(rsa_points) > 0:
            self.rsa.insert(rsa_points)
        if len(lsa_points) > 0:
            self.lsa.insert(lsa_points)
        if len(lia_points) > 0:
            self.lia.insert(lia_points)
        if len(ria_points) > 0:
            self.ria.insert(ria_points)
        if len(rsp_points) > 0:
            self.rsp.insert(rsp_points)
        if len(lsp_points) > 0:
            self.lsp.insert(lsp_points)
        if len(lip_points) > 0:
            self.lip.insert(lip_points)
        if len(rip_points) > 0:
            self.rip.insert(rip_points)

        # remove points from current node
        self.points.clear()

    def convert_to_node_arrays(self, dtype, depth=0):
        """
        Convert the quad tree to a list of nodes.
        The first list element is the root node.
        """
        node_list = []
        points_list = []

        node = np.full((1,), 0, dtype=dtype)
        node["center_x"] = self.center[0]
        node["center_y"] = self.center[1]
        node["center_z"] = self.center[2]
        node["edge_len"] = self.edge_len
        node["is_leaf"] = 1 if self.is_leaf() else 0
        node["num_points"] = len(self.points)
        node["depth"] = depth

        node_list.append(node)
        if not self.is_leaf():
            rsa_list, rsa_points = self.rsa.convert_to_node_arrays(dtype, depth + 1)
            lsa_list, lsa_points = self.lsa.convert_to_node_arrays(dtype, depth + 1)
            lia_list, lia_points = self.lia.convert_to_node_arrays(dtype, depth + 1)
            ria_list, ria_points = self.ria.convert_to_node_arrays(dtype, depth + 1)
            rsp_list, rsp_points = self.rsp.convert_to_node_arrays(dtype, depth + 1)
            lsp_list, lsp_points = self.lsp.convert_to_node_arrays(dtype, depth + 1)
            lip_list, lip_points = self.lip.convert_to_node_arrays(dtype, depth + 1)
            rip_list, rip_points = self.rip.convert_to_node_arrays(dtype, depth + 1)

            node["rsa"] = len(node_list)
            node_list.extend(rsa_list)
            node["lsa"] = len(node_list)
            node_list.extend(lsa_list)
            node["lia"] = len(node_list)
            node_list.extend(lia_list)
            node["ria"] = len(node_list)
            node_list.extend(ria_list)
            node["rsp"] = len(node_list)
            node_list.extend(rsp_list)
            node["lsp"] = len(node_list)
            node_list.extend(lsp_list)
            node["lip"] = len(node_list)
            node_list.extend(lip_list)
            node["rip"] = len(node_list)
            node_list.extend(rip_list)

            if len(rsa_points) > 0:
                points_list.extend(rsa_points)
            if len(lsa_points) > 0:
                points_list.extend(lsa_points)
            if len(lia_points) > 0:
                points_list.extend(lia_points)
            if len(ria_points) > 0:
                points_list.extend(ria_points)
            if len(rsp_points) > 0:
                points_list.extend(rsp_points)
            if len(lsp_points) > 0:
                points_list.extend(lsp_points)
            if len(lip_points) > 0:
                points_list.extend(lip_points)
            if len(rip_points) > 0:
                points_list.extend(rip_points)

        else:
            # leaf node, we need to append the points to the list of points
            node["rsa"] = -1
            node["lsa"] = -1
            node["lia"] = -1
            node["ria"] = -1
            node["rsp"] = -1
            node["lsp"] = -1
            node["lip"] = -1
            node["rip"] = -1
            points_list = self.points

        if depth == 0:
            # from root we assign offset in first_point_offset
            node_list = np.asarray(node_list)
            leafs_mask = node_list["num_points"] > 0
            cumsum = np.cumsum(node_list[leafs_mask]["num_points"])
            cumsum = np.append([0], cumsum[:-1])
            node_list["first_point_offset"][leafs_mask] = cumsum

            return node_list.reshape((-1,)),\
                np.asarray(points_list).reshape((-1, 3))

        return node_list, points_list
