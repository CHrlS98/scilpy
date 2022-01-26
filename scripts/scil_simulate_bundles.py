#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import numpy as np
import nibabel as nib
from scipy.interpolate import splprep, splev
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram
from dipy.data import get_sphere

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_config', help='JSON configuration file.')
    p.add_argument('out_tractogram', help='Output tractogram file.')

    p.add_argument('--volume_size', nargs=3, default=(10, 10, 10), type=int)
    p.add_argument('--fiber_decay', default=40, type=float,
                   help='Sharpness for a single fiber lobe.')

    add_overwrite_arg(p)
    return p


def transform_streamlines_to_vox(streamlines, volume_size):
    streamlines = copy.deepcopy(streamlines)
    # streamlines are in RASMM with origin corner
    # we need to find the translation such that the
    # most negative point goes to (0, 0, 0) and
    # a scaling such that the maximum value does not
    # exceed the volume dimensions
    min_coords = np.array([[0.0, 0.0, 0.0]])
    max_coords = np.array([[0.0, 0.0, 0.0]])
    for s in streamlines:
        # volume is (N, 3)
        min_pos = np.min(s, axis=0)
        subarr = np.array([min_pos, min_coords.squeeze()])
        min_coords = np.min(subarr, axis=0)

        max_pos = np.max(s, axis=0)
        subarr = np.array([max_pos, max_coords.squeeze()])
        max_coords = np.max(subarr, axis=0)

    # transform to voxel space
    max_coords -= min_coords
    scaling = np.min(np.array(volume_size) / max_coords)
    for s in streamlines:
        s -= min_coords
        s *= scaling

    return streamlines


def streamlines_to_segments(streamlines):
    seg_dirs = []
    seg_lengths = []
    vox_ids = []
    for s in streamlines:
        segments = s[1:] - s[:-1]
        lengths = np.linalg.norm(segments, axis=-1, keepdims=True)
        dirs = segments / lengths

        vox_ids.append(np.floor(s[:-1] + segments / 2).astype(int))
        seg_dirs.append(dirs)
        seg_lengths.append(lengths)

    seg_dirs = np.vstack(seg_dirs)
    seg_lengths = np.vstack(seg_lengths)
    vox_ids = np.vstack(vox_ids)

    print(seg_dirs.shape)
    print(seg_lengths.shape)
    print(vox_ids.shape)
    return seg_dirs, seg_lengths, vox_ids


def vonMisesFisher(x, mu, kappa):
    ck = kappa / (4.0*np.pi*np.sinh(kappa))
    return ck * np.exp(kappa * mu.dot(x.T))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_config)
    assert_outputs_exist(parser, args, args.out_tractogram)

    with open(args.in_config, 'r') as f:
        config = json.load(f)

    sphere = get_sphere('repulsion724').subdivide(2)
    mu = np.array([1, 0, 0]).reshape((1, 3))
    kappa = 40
    vmf = vonMisesFisher(sphere.vertices, mu, kappa)
    vmf /= vmf.max()  # represents a fiber element of unit length

    odf = actor.odf_slicer(vmf[None, None, :, :], sphere=sphere)
    scene = window.Scene()
    scene.add(odf)
    # window.show(scene)

    streamlines = []
    streamlines_tangent = []
    DELTA_SPLINE = 0.001
    for bundle in config['bundles']:
        centroid = np.array(bundle['centroids']).astype(float)
        up = np.array(bundle['up_vectors']).astype(float)

        tck, u = splprep(centroid.T, s=0)
        centroid_approx = np.array(splev(u, tck)).T
        centroid_offset = np.array(splev(u + DELTA_SPLINE, tck)).T

        # frenet frame
        T = centroid_offset - centroid_approx
        T /= np.linalg.norm(T, axis=-1, keepdims=True)

        up /= np.linalg.norm(up, axis=-1, keepdims=True)
        N = up - np.array([up[i].dot(T[i]) for i in range(len(up))])\
            .reshape((-1, 1))*T

        B = np.cross(T, N)

        # now that we have our frame at each point we can generate streamlines
        max_radius = bundle['radius']
        n_streamlines = bundle['n_streamlines']
        for _ in range(n_streamlines):
            radius = np.sqrt(np.random.uniform()) * max_radius
            theta = np.random.uniform() * 2.0 * np.pi
            points = centroid_approx + radius * np.cos(theta) * N\
                + radius * np.sin(theta) * B
            tck, _ = splprep(points.T, s=0)
            x_approx = np.array(splev(np.linspace(0, 1, 100), tck)).T
            streamlines.append(x_approx)

    # transform streamlines to voxel space
    streamlines = transform_streamlines_to_vox(streamlines, args.volume_size)

    header = nib.Nifti1Header()
    header.set_data_shape(args.volume_size)
    header.set_sform(np.identity(4), code='scanner')

    sft = StatefulTractogram(streamlines, reference=header,
                             space=Space.VOX,
                             origin=Origin.TRACKVIS)

    save_tractogram(sft, args.out_tractogram)

    dirs, lengths, pos = streamlines_to_segments(streamlines)
    unique_voxids = np.unique(pos, axis=0)
    print(unique_voxids)


if __name__ == '__main__':
    main()
