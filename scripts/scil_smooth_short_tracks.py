#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.spatial import KDTree

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)

import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils as track_utils

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('in_seed', help='Input seed file (.nii.gz).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')
    p.add_argument('--rng_seed', type=int, default=None)

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def unravel_irregular_grid(indices, num_pts_per_strl):
    cumsum = np.cumsum(num_pts_per_strl)
    strl_ids = np.searchsorted(cumsum, indices, 'right')
    pt_ids = np.where(strl_ids == 0, indices, indices - cumsum[strl_ids - 1])
    return strl_ids, pt_ids


def track(smooth_strl, kd_tree, search_radius, step_size,
          max_angle_cos, streamlines, num_pts_per_strl):
    curr_pos = smooth_strl[-1]
    while True:
        neighbours = kd_tree.query_ball_point(curr_pos, search_radius)

        if len(neighbours) == 0:  # no neighbours found
            break

        strl_ids, pt_ids = unravel_irregular_grid(neighbours, num_pts_per_strl)
        all_dirs = []
        all_dirs_origin = []
        for strl_id, pt_id in zip(strl_ids, pt_ids):
            strl = streamlines[strl_id]
            if pt_id > 0:
                all_dirs.append(strl[pt_id - 1] - strl[pt_id])
                all_dirs_origin.append(strl[pt_id])
            if pt_id < len(strl) - 1:
                all_dirs.append(strl[pt_id + 1] - strl[pt_id])
                all_dirs_origin.append(strl[pt_id])
        all_dirs = np.asarray(all_dirs)
        all_dirs /= np.linalg.norm(all_dirs, axis=1)[:, np.newaxis]

        # pick a direction
        if len(smooth_strl) > 1:
            prev_dir = smooth_strl[-1] - smooth_strl[-2]
            prev_dir /= np.linalg.norm(prev_dir)
        else:
            # choose a random direction
            prev_dir = all_dirs[np.random.randint(len(all_dirs))]
        weights = np.dot(all_dirs, prev_dir)
        mask = weights > max_angle_cos
        mean_dir = np.sum(all_dirs[mask] * weights[mask][:, None],
                          axis=0)
        dir_norm = np.linalg.norm(mean_dir)
        if dir_norm > 0.0:
            mean_dir = mean_dir / dir_norm * step_size
        else:
            break

        curr_pos = curr_pos + mean_dir
        smooth_strl.append(curr_pos)

    return smooth_strl


def generate_smooth_tracks(streamlines, seeds, search_radius,
                           step_size, max_angle):
    num_pts_per_strl = np.array([len(s) for s in streamlines])
    all_points = np.concatenate(streamlines, axis=0)

    max_angle_cos = np.cos(max_angle)
    kd_tree = KDTree(all_points)
    tracks = []
    valid_seeds = []
    for it, s in enumerate(seeds):
        if it % 100 == 0:
            print('Processing seed {}/{}'.format(it, len(seeds)))
        smooth_strl = [s]
        smooth_strl = track(smooth_strl, kd_tree, search_radius, step_size,
                            max_angle_cos, streamlines, num_pts_per_strl)
        smooth_strl.reverse()
        smooth_strl = track(smooth_strl, kd_tree, search_radius, step_size,
                            max_angle_cos, streamlines, num_pts_per_strl)
        if len(smooth_strl) > 1:
            tracks.append(smooth_strl)
            valid_seeds.append(s)

    return tracks, np.asarray(valid_seeds)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_seed])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    sft.to_center()  # to work with dipy seeding
    sft.to_vox()
    strl = sft.streamlines

    # change this value to increase/decrease the search radius
    search_radius = 1.0  # mm
    step_size = 0.5  # mm
    nb_seeds = 1000
    seed_per_vox = False
    max_angle = np.pi / 9.0

    vox_search_radius = search_radius / sft.voxel_sizes[0]
    vox_step_size = step_size / sft.voxel_sizes[0]

    seed_img = nib.load(args.in_seed)
    seeds = track_utils.random_seeds_from_mask(
        seed_img.get_fdata(dtype=np.float32),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.rng_seed)

    smooth_strl, seeds = generate_smooth_tracks(strl, seeds, vox_search_radius,
                                                vox_step_size, max_angle)

    # save the smoothed streamlines
    out_sft = StatefulTractogram.from_sft(smooth_strl, sft)
    out_sft.remove_invalid_streamlines()
    save_tractogram(out_sft, args.out_tractogram)

    interactive = False
    if interactive:
        scene = window.Scene()
        showm = window.ShowManager(scene, size=(800, 800),
                                   order_transparent=True)
        showm.initialize()

        lines = actor.line(strl[::2], opacity=0.1)
        some_point = actor.dots(seeds, color=(1, 1, 1))
        smooth_lines = actor.line(smooth_strl, colors=(1, 1, 1), linewidth=4)

        scene.add(some_point)
        scene.add(lines)
        scene.add(smooth_lines)
        showm.start()


if __name__ == "__main__":
    main()
