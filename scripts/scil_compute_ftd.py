#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import nibabel as nib
from scilpy.io.utils import (add_reference_arg, add_sh_basis_args,
                             assert_inputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.ftd import compute_ftd_gpu
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from fury import actor, window


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf', help='FODF image.')
    p.add_argument('in_seeds', help='Seeding mask.')
    p.add_argument('in_mask', help='Input mask.')

    p.add_argument('--npv', type=int, default=100,
                   help='Number of seeds per voxel.')
    p.add_argument('--step_size', type=float, default=0.5,
                   help='Step size in mm.')
    p.add_argument('--theta', type=float, default=20.0,
                   help='Maximum curvature angle in degrees.')
    p.add_argument('--min_length', type=float, default=5.0,
                   help='Minimum length of the streamlines.')
    p.add_argument('--max_length', type=float, default=20.0,
                   help='Maximum length of the streamlines.')

    add_sh_basis_args(p)
    add_reference_arg(p)
    return p


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

    track, ids = compute_ftd_gpu(fodf, seeds, mask,
                                 n_seeds_per_vox=args.npv,
                                 step_size=vox_step_size,
                                 theta=20.0,
                                 min_nb_points=min_nb_points,
                                 max_nb_points=max_nb_points,
                                 sh_basis=args.sh_basis)

    ids = np.asarray(ids)
    colors = np.zeros((len(ids), 3))
    colors[ids == 0] = [1.0, 0.0, 0.0]
    colors[ids == 1] = [1.0, 1.0, 0.0]
    colors[ids == 2] = [0.0, 1.0, 0.0]
    colors[ids == 3] = [0.0, 1.0, 1.0]
    colors[ids == 4] = [0.0, 0.0, 1.0]

    sphere = get_sphere('repulsion724')
    B_mat = sh_to_sf_matrix(sphere, 8, return_inv=False)

    endpoint = np.array([s[0] for s in track])
    line_a = actor.line(track, opacity=0.5)
    odf = actor.odf_slicer(fodf, sphere=sphere, B_matrix=B_mat)

    dots_a = actor.dots(endpoint, opacity=0.8, color=(1, 1, 1))
    scene = window.Scene()
    scene.add(line_a)
    scene.add(dots_a)
    scene.add(odf)
    window.show(scene)


if __name__ == '__main__':
    main()
