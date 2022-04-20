#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from scipy.spatial import KDTree

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_outputs_exist, snapshot)

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
    p.add_argument('--rng_seed', type=int, default=None,
                   help='Random number generator seed.')
    p.add_argument('--images_dir', default='debug_images',
                   help='Directory to save debug images.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def unravel_irregular_grid(indices, num_pts_per_strl):
    cumsum = np.cumsum(num_pts_per_strl)
    strl_ids = np.searchsorted(cumsum, indices, 'right')
    pt_ids = np.where(strl_ids == 0, indices, indices - cumsum[strl_ids - 1])
    return strl_ids, pt_ids


def get_closest_point_per_strl(strl_ids, pt_ids, curr_pos, streamlines):
    unique_ids = np.unique(strl_ids)
    closest_pt_ids = np.zeros(len(unique_ids), dtype=int)
    for i, curr_strl_id in enumerate(unique_ids):
        mask = strl_ids == curr_strl_id
        curr_pt_ids = pt_ids[mask]
        points = np.take_along_axis(streamlines[curr_strl_id],
                                    curr_pt_ids[:, None], axis=0)
        closest = np.argmin(np.sum((points - curr_pos)**2, axis=-1))
        closest_pt_ids[i] = curr_pt_ids[closest]
    return unique_ids, closest_pt_ids


def snapshot_debug(all_tracks, strl_dist, curr_strl,
                   curr_pos, neighbours, directory):
    # debugging and visualization
    s = window.Scene()
    if len(all_tracks) > 0:
        all_tracked_a =\
            actor.line(all_tracks,
                       colors=np.ones((len(all_tracks), 3)),
                       opacity=0.8)
        s.add(all_tracked_a)
    line_a = actor.line(strl_dist, opacity=0.2)
    curr_line_a = actor.line([curr_strl], colors=(1.0, 1.0, 1.0),
                             opacity=0.8, linewidth=2)
    curr_pos_a = actor.dots(np.array([curr_pos]))
    closest_pts_a = actor.dots(np.array(neighbours), color=(1, 1, 1))
    s.add(line_a, curr_line_a, curr_pos_a, closest_pts_a)
    snapshot(s, os.path.join(directory, 'smooth_strl_{}_{}.png'
                             .format(len(all_tracks), len(curr_strl))))


def track(smooth_strl, kd_tree, search_radius, step_size,
          max_angle_cos, streamlines, num_pts_per_strl,
          all_tracked, debug_directory):
    curr_pos = smooth_strl[-1]
    while True:
        neighbours = kd_tree.query_ball_point(curr_pos, search_radius)

        if len(neighbours) == 0:  # no neighbours found
            break

        # find all points inside a radius
        strl_ids, pt_ids = unravel_irregular_grid(neighbours, num_pts_per_strl)

        # find closest point per streamline
        unique_strl_ids, closest_pt_ids =\
            get_closest_point_per_strl(strl_ids, pt_ids, curr_pos, streamlines)

        all_dirs = []
        all_dirs_origin = []
        strl_of_interest = []
        for strl_id, pt_id in zip(unique_strl_ids, closest_pt_ids):
            strl = streamlines[strl_id]
            strl_of_interest.append(strl)
            if pt_id > 0:
                all_dirs.append(strl[pt_id - 1] - strl[pt_id])
                all_dirs_origin.append(strl[pt_id])
            if pt_id < len(strl) - 1:
                all_dirs.append(strl[pt_id + 1] - strl[pt_id])
                all_dirs_origin.append(strl[pt_id])
        all_dirs = np.asarray(all_dirs)
        all_dirs /= np.linalg.norm(all_dirs, axis=1)[:, np.newaxis]

        # debug visualization snapshot
        snapshot_debug(all_tracked, strl_of_interest,
                       smooth_strl, curr_pos,
                       all_dirs_origin, debug_directory)

        # pick a direction
        if len(smooth_strl) > 1:
            prev_dir = smooth_strl[-1] - smooth_strl[-2]
            prev_dir /= np.linalg.norm(prev_dir)
        else:
            # choose a random direction
            prev_dir = all_dirs[np.random.randint(len(all_dirs))]
        weights = np.dot(all_dirs, prev_dir)
        mask = weights > max_angle_cos

        # TODO: Replace mean direction by mean trajectory for some distance.
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
                           step_size, max_angle, debug_directory):
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
                            max_angle_cos, streamlines, num_pts_per_strl,
                            tracks, debug_directory)
        smooth_strl.reverse()
        smooth_strl = track(smooth_strl, kd_tree, search_radius, step_size,
                            max_angle_cos, streamlines, num_pts_per_strl,
                            tracks, debug_directory)
        if len(smooth_strl) > 1:
            tracks.append(np.asarray(smooth_strl))
            valid_seeds.append(s)

    return tracks, np.asarray(valid_seeds)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_seed])
    assert_outputs_exist(parser, args, [args.out_tractogram])
    assert_output_dirs_exist_and_empty(parser, args, [args.images_dir])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    sft.to_center()  # to work with dipy seeding
    sft.to_vox()
    strl = sft.streamlines

    # change this value to increase/decrease the search radius
    search_radius = 1.0  # mm
    step_size = 0.5  # mm
    nb_seeds = 1
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
                                                vox_step_size, max_angle,
                                                args.images_dir)

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
