#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import nibabel as nib
from dipy.tracking.streamlinespeed import set_number_of_points
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, assert_inputs_exist
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.ftd import compute_ftd_gpu
import matplotlib.pyplot as plt
from fury import actor, window


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf', help='FODF image.')
    p.add_argument('in_seeds', help='Seeding mask.')
    p.add_argument('in_mask', help='Input mask.')

    p.add_argument('--step-size', type=float, default=0.5,
                   help='Step size in mm.')
    p.add_argument('--min_length', type=float, default=5.0,
                   help='Minimum length of the streamlines.')
    p.add_argument('--max_length', type=float, default=20.0,
                   help='Maximum length of the streamlines.')

    add_reference_arg(p)
    return p


def _project_to_polynomial(P):
    c = np.column_stack([P[:, 0]**2,
                         P[:, 1]**2,
                         P[:, 2]**2,
                         P[:, 0]*P[:, 1],
                         P[:, 0]*P[:, 2],
                         P[:, 1]*P[:, 2],
                         P[:, 0],
                         P[:, 1],
                         P[:, 2],
                         np.ones(len(P))])
    return c


def _flip_if_needed(streamline, reference):
    d_dir = np.mean(np.sum((streamline - reference)**2, axis=-1))
    d_flip = np.mean(np.sum((streamline[::-1] - reference)**2, axis=-1))
    if d_flip < d_dir:
        streamline = streamline[::-1]
    return streamline


def _compute_ftd(streamlines, nb_points=20):
    """
    Compute the fiber trajectory distribution for a bundle.
    """
    # 1st resample the streamlines to have the same number of points
    streamlines = [set_number_of_points(s, nb_points) for s in streamlines]
    streamlines = [_flip_if_needed(s, streamlines[0]) for s in streamlines[1:]]

    # 2nd compute derivatives
    V = np.concatenate([s[1:] - s[:-1] for s in streamlines], axis=0)
    V = V / np.linalg.norm(V, axis=-1, keepdims=True)

    # 3rd compute polynomial representation of the streamlines
    C = np.concatenate([s[:-1] for s in streamlines], axis=0)
    centroid = np.mean(C, axis=0)

    # center points around centroid
    C = C - centroid
    min_bounds, max_bounds = np.min(C, axis=0), np.max(C, axis=0)

    C = _project_to_polynomial(C)

    # 4th Solve the least-squares problem to find the FTD
    FTD = np.linalg.lstsq(C, V, rcond=None)[0]

    return FTD, centroid, min_bounds, max_bounds


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_fodf, args.in_mask])

    # images
    fodf_im = nib.load(args.in_fodf)
    mask_im = nib.load(args.in_mask)
    seeds_im = nib.load(args.in_seeds)
    fodf = fodf_im.get_fdata(dtype=np.float32)
    mask = get_data_as_mask(mask_im)
    seeds = get_data_as_mask(seeds_im)

    # mm to voxel
    voxel_size = fodf_im.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    min_nb_points = int(args.min_length / args.step_size) + 1
    max_nb_points = int(args.max_length / args.step_size) + 1

    ftd = compute_ftd_gpu(fodf, seeds, mask, n_seeds_per_vox=100,
                          step_size=vox_step_size,
                          theta=20.0,
                          min_nb_points=min_nb_points,
                          max_nb_points=max_nb_points)

    line_a = actor.line(ftd)
    dots_a = actor.dots(np.concatenate(ftd, axis=0), opacity=0.2)
    scene = window.Scene()
    scene.add(line_a)
    scene.add(dots_a)
    window.show(scene)


def old_fun():
    ftd, centroid, min_bounds, max_bounds = _compute_ftd(sft.streamlines)

    # Make the grid
    xbounds = min_bounds[0], max_bounds[0]
    ybounds = min_bounds[1], max_bounds[1]
    zbounds = min_bounds[2], max_bounds[2]
    x, y, z = np.meshgrid(np.linspace(xbounds[0] - 10, xbounds[1] + 10, 20),
                          np.linspace(ybounds[0] - 10, ybounds[1] + 10, 20),
                          [0.0])

    # evaluate ftd at each point
    P = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    C = _project_to_polynomial(P)
    V = np.dot(C, ftd)

    streamlines = [s - centroid for s in sft.streamlines]
    line_a = actor.line(streamlines, opacity=0.2)
    peaks_a = actor.arrow(np.concatenate([P, P], axis=0),
                          np.concatenate([V, -V], axis=0),
                          colors=(1, 1, 1), heights=0.5)
    scene = window.Scene()
    scene.add(line_a)
    scene.add(peaks_a)
    window.show(scene)

    # ax = plt.figure().add_subplot(projection='3d')

    # ax.quiver(x, y, z, vx, vy, vz, length=0.5, normalize=True)

    # plt.show()


if __name__ == '__main__':
    main()
