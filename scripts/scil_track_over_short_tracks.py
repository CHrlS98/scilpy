#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete streamlines from short-tracks tractogram.
"""
import argparse
import logging

from time import perf_counter

import numpy as np
import nibabel as nib

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
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager


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
    flat_index = np.zeros(len(indices), dtype=np.int32)
    for i, id in enumerate(indices):
        flat_index[i] = (id[2] * dims[1] * dims[0] +
                         id[1] * dims[0] + id[0])
    return flat_index


@jit(nopython=True, cache=True)
def create_accel_struct_from_seeds(seed_pts, edge_length):
    """
    Create an acceleration structure for searching nearest
    streamlines on the GPU using a regular grid.

    Parameters
    ----------
    seed_pts : ndarray
        Array of streamline seed points.
    edge_length : float
        Edge length of the regular grid bins.
    """
    seed_int = (seed_pts / edge_length).astype(np.int32)
    dims = (np.max(seed_int[:, 0]) + 1,
            np.max(seed_int[:, 1]) + 1,
            np.max(seed_int[:, 2]) + 1)

    # ici j'ai les id de toutes mes cellules non vides, triées
    cell_ids = np.unique(ravel_multi_index(seed_int, dims))

    # pour chaque streamline, on génère les strl_counts
    # nombre de streamlines qui traverse chaque cellule de la grille
    cell_strl_count = np.zeros(len(cell_ids), dtype=np.int32)
    for seed_i in range(len(seed_int)):
        curr_cell_id = ravel_multi_index(seed_int[seed_i:seed_i + 1], dims)
        pos = dichotomic_search(curr_cell_id, cell_ids)
        cell_strl_count[pos] += 1

    # offsets in streamlines ids array
    cell_strl_offsets = np.append([0], np.cumsum(cell_strl_count)[:-1])

    # on va devoir repasser à travers toutes les streamlines
    # pour generer le tableau de streamline ids
    cell_strl_ids = np.full(np.sum(cell_strl_count), -1, dtype=np.int32)

    for strl_id in range(len(seed_int)):
        curr_cell_id = ravel_multi_index(seed_int[strl_id:strl_id + 1], dims)

        pos = dichotomic_search(curr_cell_id, cell_ids)
        cell_strl_offset = cell_strl_offsets[pos]
        emplace_pos = 0
        while cell_strl_ids[cell_strl_offset + emplace_pos] >= 0:
            emplace_pos += 1
        cell_strl_ids[cell_strl_offset + emplace_pos] = strl_id

    return cell_ids, cell_strl_count, cell_strl_offsets, cell_strl_ids, dims


@jit(nopython=True, cache=True)
def create_accel_struct(strl_pts, strl_lengths, edge_length):
    """
    Create an acceleration structure for searching nearest
    streamlines on the GPU using a regular grid.

    Parameters
    ----------
    strl_pts : ndarray
        Array of streamline points.
    strl_lengths : ndarray
        Array of streamline lengths.
    edge_length : float
        Edge length of the regular grid bins.
    """
    strl_int = (strl_pts / edge_length).astype(np.int32)
    dims = (np.max(strl_int[:, 0]) + 1,
            np.max(strl_int[:, 1]) + 1,
            np.max(strl_int[:, 2]) + 1)

    # ici j'ai les id de toutes mes cellules non vides, triées
    cell_ids = np.unique(ravel_multi_index(strl_int, dims))

    # pour chaque streamline, on génère les strl_counts
    # nombre de streamlins qui traverse chaque cellule de la grille
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
    # st_seeds are (vox, center) but we want them (vox, corner)
    st_seeds = sft.data_per_streamline['seeds'] + 0.5

    logging.info('Loaded tractogram containing {0} streamlines in {1:.2f}s.'
                 .format(len(sft.streamlines), perf_counter() - t0))

    vox_search_radius = args.search_radius / sft.voxel_sizes[0]
    vox_step_size = args.step_size / sft.voxel_sizes[0]
    min_cos_angle = float(np.cos(np.deg2rad(args.theta)))
    min_cos_angle_init = float(np.cos(np.deg2rad(args.theta_init)))
    max_strl_len = int(args.max_length / args.step_size) + 1
    min_strl_len = int(args.min_length / args.step_size) + 1

    st_pts = np.concatenate(sft.streamlines, axis=0)
    st_barycenters = [np.mean(s, axis=0) for s in sft.streamlines]

    st_lengths = sft.streamlines._lengths
    st_offsets = np.append([0], np.cumsum(st_lengths)[:-1])

    # create acceleration structure around streamline points
    t0 = perf_counter()
    cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids, grid_dims =\
        create_accel_struct_from_seeds(st_seeds, st_lengths, vox_search_radius)
    min_density = int(np.min(cell_st_counts))
    max_density = int(np.max(cell_st_counts))
    logging.info('Created acceleration structure in {0:.2f}s.'
                 .format(perf_counter() - t0))
    logging.info('Structure contains {0} cells. Min(max) density '
                 ' is {1}({2}). Mean density is {3:.2f}.'
                 .format(len(cell_ids), min_density, max_density,
                         np.mean(cell_st_counts)))

    from fury import window, actor
    s = window.Scene()
    strl_viz = [sft.streamlines[s] for s in cell_st_ids[:8]]
    line = actor.line(strl_viz)
    s.add(line)
    window.show(s)

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
    # seed_pts = np.tile(seed_pts, (5, 1))
    nb_seeds = len(seed_pts)
    logging.info('Generated {0} seed positions in {1:.2f}s.'
                 .format(nb_seeds, perf_counter() - t0))

    # prepare arrays for gpu
    cell_ids = np.asarray(cell_ids, dtype=np.int32)
    cell_st_counts = np.asarray(cell_st_counts, dtype=np.int32)
    cell_st_offsets = np.asarray(cell_st_offsets, dtype=np.int32)
    cell_st_ids = np.asarray(cell_st_ids, dtype=np.int32)
    st_lengths = np.asarray(st_lengths, dtype=np.int32)
    st_offsets = np.asarray(st_offsets, dtype=np.int32)
    st_pts = np.column_stack((st_pts, np.ones(len(st_pts)))).flatten()\
        .astype(np.float32)
    st_barycenters = np.asarray(st_barycenters, dtype=np.float32).flatten()
    seed_pts = np.column_stack((seed_pts, np.ones(len(seed_pts)))).flatten()\
        .astype(np.float32)

    cl_kernel = CLKernel('track_over_tracks', 'tracking',
                         'track_over_tracks_v2.cl')

    # add compiler definitions
    cl_kernel.set_define('NUM_CELLS', f'{len(cell_ids)}')
    cl_kernel.set_define('SEARCH_RADIUS', f'{vox_search_radius:.5}f')
    # cl_kernel.set_define('EDGE_LENGTH', f'{edge_length:.5}f')
    cl_kernel.set_define('MAX_DENSITY', f'{max_density}')
    cl_kernel.set_define('MAX_SEARCH_NEIGHBOURS', '27')
    cl_kernel.set_define('CELLS_XMAX', f'{grid_dims[0]}')
    cl_kernel.set_define('CELLS_YMAX', f'{grid_dims[1]}')
    cl_kernel.set_define('CELLS_ZMAX', f'{grid_dims[2]}')
    cl_kernel.set_define('MAX_STRL_LEN', f'{max_strl_len}')
    cl_kernel.set_define('MIN_COS_ANGLE', f'{min_cos_angle:.5}f')
    cl_kernel.set_define('MIN_COS_ANGLE_INIT', f'{min_cos_angle_init:.5}f')
    cl_kernel.set_define('STEP_SIZE', f'{vox_step_size}f')

    cl_manager = CLManager(cl_kernel, n_inputs=9, n_outputs=2)

    # inputs
    cl_manager.add_input_buffer(0, cell_ids, dtype=cell_ids.dtype)
    cl_manager.add_input_buffer(1, cell_st_counts, dtype=cell_st_counts.dtype)
    cl_manager.add_input_buffer(2, cell_st_offsets, dtype=cell_st_offsets.dtype)
    cl_manager.add_input_buffer(3, cell_st_ids, dtype=cell_st_ids.dtype)
    cl_manager.add_input_buffer(4, st_lengths, dtype=st_lengths.dtype)
    cl_manager.add_input_buffer(5, st_offsets, dtype=st_offsets.dtype)
    cl_manager.add_input_buffer(6, st_pts, dtype=st_pts.dtype)
    cl_manager.add_input_buffer(7, st_barycenters, dtype=st_barycenters.dtype)
    cl_manager.add_input_buffer(8, seed_pts, dtype=seed_pts.dtype)

    # outputs
    cl_manager.add_output_buffer(0, (nb_seeds*max_strl_len*4,),
                                 dtype=np.float32)
    cl_manager.add_output_buffer(1, (nb_seeds,), dtype=np.int32)

    t0 = perf_counter()
    logging.info(f'Launching tracking...')
    output_tracks, output_tracks_len = cl_manager.run((nb_seeds, 1, 1))
    output_tracks = np.asarray(output_tracks).reshape((-1, 4))

    strl = []
    for i in range(nb_seeds):
        num_pts = output_tracks_len[i]
        if(num_pts >= min_strl_len):
            strl_pts = output_tracks[i*max_strl_len:i*max_strl_len+num_pts]
            strl.append(strl_pts[..., :-1])
    logging.info(f'Tracking finished in {perf_counter() - t0:.2f}s.')

    logging.info('Saving output tractogram.')
    out_sft = StatefulTractogram.from_sft(strl, sft)
    out_sft.remove_invalid_streamlines()
    save_tractogram(out_sft, args.out_tractogram)

    logging.info('Total runtime: {:.2f}s.'.format(perf_counter() - t_init))


if __name__ == '__main__':
    main()
