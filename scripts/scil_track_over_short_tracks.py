#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete streamlines from short-tracks tractogram.
"""
import argparse
import inspect
import scilpy
import os
import numpy as np
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_reference_arg, add_overwrite_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
import pyopencl as cl
from numba import jit


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('in_seed', help='Input seed file (.nii.gz).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    p.add_argument('--search_radius', type=float, default=0.5,
                   help='Search radius in mm. [%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


@jit(nopython=True, cache=True)
def dichotomic_search(value, array):
    num_values = len(array)
    search_range = (0, num_values - 1)

    keep_searching = True
    while keep_searching:
        if search_range[1] - search_range[0] > 1:
            mid_id = (search_range[0] + search_range[1]) // 2
            if value == array[mid_id]:
                return mid_id
            elif value < array[mid_id]:
                search_range = (search_range[0], mid_id)
            else:
                search_range = (mid_id, search_range[1])
        else:
            if value == array[search_range[0]]:
                return search_range[0]
            elif value == array[search_range[1]]:
                return search_range[1]
            else:
                return -1
    return -1


def search_neighbours(point, cell_ids, strl_per_cell,
                      num_strl_per_cell, dims, vox_radius):
    grid_id = (point / vox_radius).astype(int)
    flat_id = ravel_multi_index(grid_id, dims)
    # find position in flat ids array
    # 1. can be in the array
    # 2. can also not be in the array (meaning the cell is empty)
    offsets = np.append([0], np.cumsum(num_strl_per_cell)[:-1])
    unique_id = dichotomic_search(flat_id, cell_ids)
    if unique_id >= 0:
        # neighbours found!
        num_candidate_strls = num_strl_per_cell[unique_id]
        first_point_id = offsets[unique_id]
        return strl_per_cell[first_point_id:first_point_id+num_candidate_strls]
    return unique_id


@jit(nopython=True, cache=True)
def ravel_multi_index(indices, dims):
    # print("RAVEL MULTI INDEX")
    flat_index = np.zeros(len(indices), dtype=np.int32)
    for i, id in enumerate(indices):
        flat_index[i] = (id[2] * dims[1] * dims[0] +
                         id[1] * dims[0] + id[0])
    return flat_index


@jit(nopython=True, cache=True)
def generate_streamline_ids(strl_lengths):
    print("GENERATE STRL IDS")
    cumsum = np.cumsum(strl_lengths)
    ids = np.zeros(cumsum[-1], dtype=np.int32)
    start = 0
    for i, length in enumerate(strl_lengths):
        stop = start + length
        ids[start:stop] = i
        start = stop
    return ids


@jit(nopython=True, cache=True)
def create_accel_struct(strl_pts, strl_lengths, edge_length):
    strl_int = (strl_pts / edge_length).astype(np.int32)
    dims = (np.max(strl_int[:, 0]) + 1,
            np.max(strl_int[:, 1]) + 1,
            np.max(strl_int[:, 2]) + 1)

    # ici j'ai les id de toutes mes cellules non vides, triées
    cell_ids = np.unique(ravel_multi_index(strl_int, dims))

    # pour chaque streamline, on génère les strl_counts
    offset = 0
    cell_strl_count = np.zeros(len(cell_ids), dtype=np.int32)
    for length in strl_lengths:
        curr_strl_cell_ids = ravel_multi_index(
            strl_int[offset:offset+length], dims)
        unique = set(curr_strl_cell_ids)
        for u in unique:
            pos = dichotomic_search(u, cell_ids)
            cell_strl_count[pos] += 1
        offset += length

    # offsets in streamlines ids array
    cell_strl_offsets = np.append([0], np.cumsum(cell_strl_count)[:-1])

    # on va devoir repasser à travers toutes les streamlines
    # pour generer le tableau de streamline ids
    cell_strl_ids = np.full(np.sum(cell_strl_count), -1, dtype=np.int32)
    offset = 0
    for strl_id, length in enumerate(strl_lengths):
        curr_strl_cell_ids = ravel_multi_index(
            strl_int[offset:offset+length], dims)
        unique = set(curr_strl_cell_ids)

        for u in unique:
            pos = dichotomic_search(u, cell_ids)
            cell_strl_offset = cell_strl_offsets[pos]
            emplace_pos = 0
            while cell_strl_ids[cell_strl_offset + emplace_pos] >= 0:
                emplace_pos += 1
            cell_strl_ids[cell_strl_offset + emplace_pos] = strl_id
        offset += length

    return cell_ids, cell_strl_count, cell_strl_offsets, cell_strl_ids, dims


def get_cl_code_string(fpath):
    f = open(fpath, 'r')
    code_str = f.read()
    f.close()

    # optionally remove code above the //$TRIMABOVE$ mark,
    # useful for replacing define and structs inside file
    trim = code_str.find("//$TRIMABOVE$")
    if trim > 0:
        code_str = code_str[trim:]
    return code_str


def main():
    # p = psutil.Process()
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_seed])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()

    vox_search_radius = args.search_radius / sft.voxel_sizes[0]
    edge_length = vox_search_radius
    max_search_neighbours = int(vox_search_radius / edge_length + 1)**3

    st_pts = np.concatenate(sft.streamlines, axis=0)
    min_position = np.min(st_pts, axis=0)
    st_pts -= min_position  # première valeur à (0, 0, 0) -> bounding box plus tight
    st_lengths = sft.streamlines._lengths
    st_offsets = np.append([0], np.cumsum(st_lengths)[:-1])

    cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids, grid_dims =\
        create_accel_struct(st_pts, st_lengths, edge_length)
    seed_pts = np.array([[41.0, 30.0, 1.0]])

    # print(p.memory_full_info())

    print("Number of bins:", len(cell_ids))
    print("Max density:", np.max(cell_st_counts))

    # prepare arrays for gpu
    cell_ids_asarray = np.asarray(cell_ids, dtype=np.int32)
    cell_st_counts_asarray = np.asarray(cell_st_counts, dtype=np.int32)
    cell_st_offsets_asarray = np.asarray(cell_st_offsets, dtype=np.int32)
    cell_st_ids_asarray = np.asarray(cell_st_ids, dtype=np.int32)
    st_lengths_asarray = np.asarray(st_lengths, dtype=np.int32)
    st_offsets_asarray = np.asarray(st_offsets, dtype=np.int32)
    st_pts_asarray = np.column_stack((st_pts, np.ones(len(st_pts))))\
        .astype(np.float32)
    seed_pts_asarray = np.column_stack((seed_pts, np.ones(len(seed_pts))))\
        .astype(np.float32)

    # output buffer
    out_tracks = np.zeros((max_search_neighbours, 4), dtype=np.float32)

    # create context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # copy arrays to device (gpu)
    cell_ids_dev = cl.array.to_device(queue, cell_ids_asarray)
    cell_st_counts_dev = cl.array.to_device(queue, cell_st_counts_asarray)
    cell_st_offsets_dev = cl.array.to_device(queue, cell_st_offsets_asarray)
    cell_st_ids_dev = cl.array.to_device(queue, cell_st_ids_asarray)
    st_lengths_dev = cl.array.to_device(queue, st_lengths_asarray)
    st_offsets_dev = cl.array.to_device(queue, st_offsets_asarray)
    st_pts_dev = cl.array.to_device(queue, st_pts_asarray)
    seed_pts_dev = cl.array.to_device(queue, seed_pts_asarray)

    # read CL kernel code
    module_path = inspect.getfile(scilpy)
    cl_code_path = os.path.join(os.path.dirname(module_path),
                                'tracking', 'track_over_tracks.cl')
    code_str = get_cl_code_string(cl_code_path)

    # add compiler definitions

    print(0)


if __name__ == '__main__':
    main()
