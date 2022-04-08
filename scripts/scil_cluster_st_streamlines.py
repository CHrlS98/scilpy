#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import cluster

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)

from dipy.data import get_sphere
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def shape_dist(s, t):
    ds = s[1:] - s[:-1]
    dt = t[1:] - t[:-1]
    dt_rev = dt[::-1]

    dist = np.sum(np.sum((ds - dt)**2, axis=1))
    dist_rev = np.sum(np.sum((ds - dt_rev)**2, axis=1))
    if dist < dist_rev:
        return dist
    return dist_rev


def spatial_dist(s, t):
    t_rev = t[::-1]

    return min(np.sum(np.sum((s - t)**2, axis=1)),
               np.sum(np.sum((s - t_rev)**2, axis=1)))


def reorder_streamlines(streamlines, n_epochs=10):
    """
    The midpoint of each streamline is inside the same voxel.
    """
    mid_pos = streamlines.shape[1] // 2
    fwd_dir = streamlines[:, mid_pos + 1] - streamlines[:, mid_pos]
    fwd_dir /= np.linalg.norm(fwd_dir, axis=1)[:, np.newaxis]

    pdir = fwd_dir[np.random.randint(len(streamlines))]
    for _ in range(n_epochs):
        aligned = np.dot(fwd_dir, pdir) > 0.0
        pdir = np.sum(fwd_dir[aligned], axis=0)
        pdir /= np.linalg.norm(pdir)

    flip_mask = np.dot(fwd_dir, pdir) < 0.0
    streamlines[flip_mask] =\
        streamlines[flip_mask][np.arange(np.count_nonzero(flip_mask)), ::-1]
    return streamlines


def streamlines_kmeans(streamlines, nb_clusters=2, n_epochs=50):
    """
    Perform kmeans in streamlines space.

    Because a streamline and the reverse of a streamline represent the same
    trajectory, we duplicate each trajectory to account for the streamline
    and its inverse.

    How many clusters?
    """
    strl_arr = np.array([s for s in streamlines])

    cluster_id = np.random.randint(nb_clusters, size=len(strl_arr))
    cluster_means = np.zeros((nb_clusters,) + strl_arr.shape[1:])
    for _ in range(n_epochs):
        for i in range(nb_clusters):
            mask = cluster_id == i
            if np.count_nonzero(mask) > 0:
                s_in_cluster = strl_arr[mask]
                s_in_cluster = reorder_streamlines(s_in_cluster)
                strl_arr[mask] = s_in_cluster
                cluster_means[i] = np.mean(s_in_cluster, axis=0)

        updated_clusters = np.zeros_like(cluster_id)
        for idx, s in enumerate(strl_arr):
            dists = np.zeros((nb_clusters,))
            for i in range(nb_clusters):
                dists[i] = spatial_dist(s, cluster_means[i])
            updated_clusters[idx] = np.argmin(dists)

        cluster_id = updated_clusters

    means = np.array([np.mean(strl_arr[cluster_id == i], axis=0)
                      for i in range(nb_clusters)])
    return cluster_id, means


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    sft.to_corner()
    sft.to_vox()
    strl = sft.streamlines

    seeds_vox = sft.data_per_streamline['seeds']
    seeds_vox += 0.5  # to origin corner
    seeds_vox = seeds_vox.astype(int)  # voxel ID

    unique_vox = np.unique(seeds_vox, axis=0)
    nb_clusters = 2
    all_clusters = []
    all_means = []
    for vox in unique_vox:
        clusters, means =\
            streamlines_kmeans(strl[np.all(vox == seeds_vox, axis=1)],
                               nb_clusters)
        all_clusters.append(clusters)
        all_means.append(means)

    all_clusters = np.concatenate(all_clusters, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    print(all_clusters.shape)
    print(all_means.shape)

    # Save cluster means
    out_sft = StatefulTractogram.from_sft(all_means, sft)
    save_tractogram(out_sft, args.out_tractogram)

    line_actor = actor.line(strl, colors=clusters)
    means_actor = actor.line(all_means, colors=(1, 1, 1))
    s = window.Scene()
    # s.add(line_actor)
    s.add(means_actor)
    window.show(s)


if __name__ == "__main__":
    main()
