#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import nibabel as nib
from scilpy.io.utils import (add_reference_arg, add_sh_basis_args,
                             assert_inputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.ftd import FTDFitter
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from fury import actor, window


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf', help='FODF image.')
    p.add_argument('in_seeds', help='Seeding mask.')
    p.add_argument('in_mask', help='Input mask.')
    p.add_argument('out_ftd', help='Output FTD file (json).')

    p.add_argument('--npv', type=int, default=100,
                   help='Number of seeds per voxel.')
    p.add_argument('--step_size', type=float, default=0.5,
                   help='Step size in mm.')
    p.add_argument('--theta', type=float, default=20.0,
                   help='Maximum curvature angle in degrees.')
    p.add_argument('--angular_mdf_threshold', type=float, default=-0.9,
                   help='Angular minimum average direct-flip distance'
                        ' threshold [%(default)s]')
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

    ftd_fitter = FTDFitter(fodf, seeds, mask, args.npv, vox_step_size,
                           args.theta, min_nb_points, max_nb_points,
                           sh_basis=args.sh_basis)

    ftd, track, ids = ftd_fitter.fit()
    ftd.save_to_json('ftd.json')


if __name__ == '__main__':
    main()
