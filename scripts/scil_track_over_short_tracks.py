#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete streamlines from short-tracks tractogram.
"""
import argparse
import inspect
import logging
import scilpy
import os
from time import perf_counter

import numpy as np
import nibabel as nib
import pyopencl as cl

from pyopencl import array
from numba import jit
from dipy.tracking.utils import random_seeds_from_mask
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist, add_reference_arg,
                             add_overwrite_arg)
from scilpy.tracking.utils import add_seeding_options
from scilpy.io.streamlines import load_tractogram_with_reference


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('in_seed', help='Input seed file (.nii.gz).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    p.add_argument('--search_radius', type=float, default=1.0,
                   help='Search radius in mm. [%(default)s]')
    p.add_argument('--bbox_edge_length', type=float, default=1.0,
                   help='Edge length of bounding box cells. [%(default)s]')

    add_seeding_options(p)
    p.add_argument('--step_size', type=float, default=0.5,
                   help='Step size in mm. [%(default)s]')
    p.add_argument('--theta', type=float, default=20.0,
                   help='Maximum angle between 2 steps. If more than one value'
                        '\nare given, the maximum angle will be drawn at '
                        'random\nfrom the distribution for each streamline. '
                        '[%(default)s]')
    p.add_argument('--theta_init', type=float, default=60.0,
                   help='Maximum deviation angle for selecting the '
                        'first tracking direction. [%(default)s]')
    p.add_argument('--min_length', type=float, default=20.0,
                   help='Minimum length of the streamline '
                        'in mm. [%(default)s]')
    p.add_argument('--max_length', type=float, default=300.0,
                   help='Maximum length of the streamline '
                        'in mm. [%(default)s]')
    p.add_argument('--rand', type=int, default=1234,
                   help='Random seed for tracking seeds. [%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


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


@jit(nopython=True, cache=True)
def ravel_multi_index(indices, dims):
    # print("RAVEL MULTI INDEX")
    flat_index = np.zeros(len(indices), dtype=np.int32)
    for i, id in enumerate(indices):
        flat_index[i] = (id[2] * dims[1] * dims[0] +
                         id[1] * dims[0] + id[0])
    return flat_index


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


def main():
    t_init = perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram, args.in_seed])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    t0 = perf_counter()
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()
    logging.info('Loaded tractogram containing {0} streamlines in {1:.2}s.'
                 .format(len(sft.streamlines), perf_counter() - t0))

    vox_search_radius = args.search_radius / sft.voxel_sizes[0]
    vox_step_size = args.step_size / sft.voxel_sizes[0]
    edge_length = args.bbox_edge_length / sft.voxel_sizes[0]
    max_search_neighbours = int(2 * vox_search_radius / edge_length + 1)**3
    min_cos_angle = float(np.cos(np.deg2rad(args.theta)))
    min_cos_angle_init = float(np.cos(np.deg2rad(args.theta_init)))
    max_strl_len = int(args.max_length / args.step_size) + 1
    min_strl_len = int(args.min_length / args.step_size) + 1

    st_pts = np.concatenate(sft.streamlines, axis=0)
    min_position = np.min(st_pts, axis=0)
    st_pts -= min_position  # bounding box plus tight

    st_lengths = sft.streamlines._lengths
    st_offsets = np.append([0], np.cumsum(st_lengths)[:-1])

    # create acceleration structure around streamline points
    t0 = perf_counter()
    cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids, grid_dims =\
        create_accel_struct(st_pts, st_lengths, edge_length)
    max_density = int(np.max(cell_st_counts))
    logging.info('Created acceleration structure in {0:.2}s.'
                 .format(perf_counter() - t0))

    # generate seeds
    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    t0 = perf_counter()
    mask = get_data_as_mask(nib.load(args.in_seed))
    seed_pts = random_seeds_from_mask(
        mask, np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.rand)
    seed_pts -= min_position
    logging.info('Generated {0} seed positions in {1:.2f}s.'
                 .format(len(seed_pts), perf_counter() - t0))

    # prepare arrays for gpu
    cell_ids = np.asarray(cell_ids, dtype=np.int32)
    cell_st_counts = np.asarray(cell_st_counts, dtype=np.int32)
    cell_st_offsets = np.asarray(cell_st_offsets, dtype=np.int32)
    cell_st_ids = np.asarray(cell_st_ids, dtype=np.int32)
    st_lengths = np.asarray(st_lengths, dtype=np.int32)
    st_offsets = np.asarray(st_offsets, dtype=np.int32)
    st_pts = np.column_stack((st_pts, np.ones(len(st_pts)))).astype(np.float32)
    seed_pts = np.column_stack((seed_pts, np.ones(len(seed_pts))))\
        .astype(np.float32)

    # output buffer
    out_tracks = np.zeros((max_strl_len * len(seed_pts), 4), dtype=np.float32)

    # create context and command queue
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # copy arrays to device (gpu)
    cell_ids_dev = cl.array.to_device(queue, cell_ids)
    cell_st_counts_dev = cl.array.to_device(queue, cell_st_counts)
    cell_st_offsets_dev = cl.array.to_device(queue, cell_st_offsets)
    cell_st_ids_dev = cl.array.to_device(queue, cell_st_ids)
    st_lengths_dev = cl.array.to_device(queue, st_lengths)
    st_offsets_dev = cl.array.to_device(queue, st_offsets)
    st_pts_dev = cl.array.to_device(queue, st_pts)
    seed_pts_dev = cl.array.to_device(queue, seed_pts)
    out_tracks_dev = cl.array.to_device(queue, out_tracks)

    # read CL kernel code
    module_path = inspect.getfile(scilpy)
    cl_code_path = os.path.join(os.path.dirname(module_path),
                                'tracking', 'track_over_tracks.cl')
    code_str = get_cl_code_string(cl_code_path)

    # add compiler definitions
    num_cells_str = f'#define NUM_CELLS {len(cell_ids)}\n'
    search_rad_str = f'#define SEARCH_RADIUS {vox_search_radius:.5}f\n'
    edge_len_str = f'#define EDGE_LENGTH {edge_length:.5}f\n'
    max_density_str = f'#define MAX_DENSITY {max_density}\n'
    max_search_neighbours_str =\
        f'#define MAX_SEARCH_NEIGHBOURS {max_search_neighbours}\n'
    xmax_str = f'#define XMAX {grid_dims[0]}\n'
    ymax_str = f'#define YMAX {grid_dims[1]}\n'
    zmax_str = f'#define ZMAX {grid_dims[2]}\n'
    max_strl_len_str = f'#define MAX_STRL_LEN {max_strl_len}\n'
    min_cos_angle_str = f'#define MIN_COS_ANGLE {min_cos_angle:.5}f\n'
    min_cos_angle_init_str =\
        f'#define MIN_COS_ANGLE_INIT {min_cos_angle_init:.5}f\n'
    vox_step_size_str = f'#define STEP_SIZE {vox_step_size}f\n'

    code_str =\
        num_cells_str + search_rad_str + edge_len_str +\
        max_density_str + max_search_neighbours_str + xmax_str +\
        ymax_str + zmax_str + max_strl_len_str + min_cos_angle_str +\
        min_cos_angle_init_str + vox_step_size_str + code_str

    # create program
    program = cl.Program(ctx, code_str).build()

    t0 = perf_counter()
    logging.info(f'Launching tracking for {seed_pts_dev.shape[:-1]} '
                 'seed positions.')
    evt = program.track_over_tracks(
        queue, seed_pts_dev.shape[:1], None, cell_ids_dev.data,
        cell_st_counts_dev.data, cell_st_offsets_dev.data,
        cell_st_ids_dev.data, st_lengths_dev.data, st_offsets_dev.data,
        st_pts_dev.data, seed_pts_dev.data, out_tracks_dev.data)

    evt.wait()
    logging.info(f'Tracking finished in {perf_counter() - t0:.2f}s.')

    output_tracks = np.asarray(out_tracks_dev.get())
    strl = []
    for i in range(seed_pts_dev.shape[0]):
        strl_pts = output_tracks[i*max_strl_len:i*max_strl_len+max_strl_len]
        strl_pts = strl_pts[strl_pts[..., -1] > 0][..., :-1] + min_position
        if(len(strl_pts >= min_strl_len)):
            strl.append(strl_pts)

    # save output tractogram
    logging.info('Saving output tractogram.')
    out_sft = StatefulTractogram.from_sft(strl, sft)
    out_sft.remove_invalid_streamlines()
    save_tractogram(out_sft, args.out_tractogram)

    logging.info('Total runtime: {:.2f}s.'.format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
