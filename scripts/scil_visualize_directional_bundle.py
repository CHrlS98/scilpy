#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import argparse
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)

from dipy.data import get_sphere
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from sklearn.decomposition import PCA
from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def compute_main_axes(strl_arr):
    """
    Compute the main axes of the streamlines array.
    """
    point_cloud = np.reshape(strl_arr, (-1, 3))
    pca = PCA()
    pca.fit(point_cloud)
    main_axes = pca.components_
    return main_axes


def compute_forward_dir_per_streamline(seeds, strl_arr):
    seed_idx = []
    for i, seed_pos in enumerate(seeds):
        idx = np.sum(np.abs(strl_arr[i] - seed_pos), axis=-1).argmin()
        seed_idx.append(idx)

    forward_dir =\
        strl_arr[np.arange(len(strl_arr)), np.asarray(seed_idx) + 1, :] -\
        strl_arr[np.arange(len(strl_arr)), seed_idx, :]
    return forward_dir


def compute_mean_streamline_per_seed(sft):
    """
    Compute the mean trajectory per seed.
    """
    seeds = sft.data_per_streamline['seeds']
    strl_arr = np.array([s for s in sft.streamlines])
    forward_dir = compute_forward_dir_per_streamline(seeds, strl_arr)

    seed_vox = (seeds + 0.5).astype(int)

    mean_streamlines = []
    main_axes_list = []
    vox_centers_list = []
    for vox in np.unique(seed_vox, axis=0):
        submask = np.all(seed_vox == vox, axis=1)
        subarr = strl_arr[submask]
        subdirs = forward_dir[submask]
        print(vox)
        print(subarr.shape)
        print(subdirs.shape)

        main_axes = compute_main_axes(subarr)
        main_axes_list.append(main_axes)
        vox_centers_list.append([vox, vox, vox])

        cos_theta = np.stack((np.abs(main_axes[0].dot(subdirs.T)),
                              np.abs(main_axes[1].dot(subdirs.T)),
                              np.abs(main_axes[2].dot(subdirs.T))), axis=1)

        # the main axis along which each streamline is best aligned
        best_align = np.argmax(cos_theta, axis=1)
        for i in range(3):
            mask = subdirs[best_align == i].dot(main_axes[i]) < 0.0
            subsubarr = subarr[best_align == i]
            if np.count_nonzero(mask) > 0:
                subsubarr[mask] =\
                    subsubarr[mask][np.arange(np.count_nonzero(mask)), ::-1]
                mean_streamlines.append(np.mean(subsubarr, axis=0))

    return mean_streamlines, main_axes_list, vox_centers_list


def distance(s, t):
    """
    Compute the distance between two equal length streamlines.
    """
    assert s.shape == t.shape, "s and t must have the same shape"
    dist = min(1.0/len(s)*np.sum(np.sqrt(np.sum((s - t)**2, axis=-1))),
               1.0/len(s)*np.sum(np.sqrt(np.sum((s - t[::-1])**2, axis=-1))))
    return dist


def kmeans_for_matrices(streamlines, nb_clusters=3):
    """
    Perform kmeans in streamlines space.
    """
    cluster_id = np.random.randint(nb_clusters, size=len(streamlines))
    for i in range(nb_clusters):
        mask = cluster_id == i
        if np.count_nonzero(mask) > 0:
            s_in_cluster = streamlines[mask]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_center()
    sft.to_vox()

    kmeans_for_matrices(sft.streamlines)

    # seeds are in vox space with origin center
    seeds = sft.data_per_streamline['seeds']

    mean_strl, main_axes, vox_centers = compute_mean_streamline_per_seed(sft)

    scene = window.Scene()
    scene.add(actor.line(mean_strl, colors=(1, 1, 1), linewidth=6.0))
    line_actor = actor.line(sft.streamlines, linewidth=2.0, opacity=0.2)
    seed_actor = actor.dots(seeds, color=(1, 1, 1))
    colors = np.tile([[1, 0, 0], [0, 1, 0], [0, 0, 1]], len(main_axes)).reshape((-1, 3))
    axes_actor = actor.arrow(np.asarray(vox_centers).reshape((-1, 3)),
                             np.asarray(main_axes).reshape((-1, 3)),
                             colors)
    scene.add(line_actor)
    scene.add(seed_actor)
    scene.add(axes_actor)
    window.show(scene)


if __name__ == "__main__":
    main()
