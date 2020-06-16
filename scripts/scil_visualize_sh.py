#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize spherical harmonics loaded from NIfTI1 image
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from fury.data import read_viz_icons, fetch_viz_icons
from fury import window, actor, ui

from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args)

WINDOW_SIZE=(512, 512)
slice_index = 50

def visualize_sfs(sfs, sphere, scale=0.8, out_path=None, window_size=WINDOW_SIZE):
    odfs = actor.odf_slicer(sfs, radial_scale=True,
                            sphere=sphere, colormap='jet', scale=scale)

    scene = window.Scene()
    showm = window.ShowManager(scene, size=window_size, title='Visualize SH')
    showm.initialize()

    scene.add(odfs)

    showm.start()

    if out_path:
        window.record(scene, size=WINDOW_SIZE, out_path=out_path, reset_camera=False)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('todi_sh_filename',
                   help='Input SH image file.')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--axis_name', default='axial',
                   help='Name of the axis to visualize.\
                   One of: [sagittal, coronal, axial]')

    p.add_argument('--output', help='Path of output file.')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH.')

    add_sh_basis_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print('Executing scil_visualize_sh script')
    img = nib.nifti1.load(args.todi_sh_filename)
    print('Loaded NIfTI1 image')
    data = img.get_fdata()

    data_slice = None
    if args.axis_name == 'sagittal':
        data_slice = data[args.slice_index, :, :, :]
    elif args.axis_name == 'coronal':
        data_slice = data[:, args.slice_index, :, :]
    else:
        data_slice = data[:, :, args.slice_index, :]

    sph_gtab = get_sphere('symmetric724')

    sfs = sh_to_sf(data_slice[:, :, None, :], sph_gtab, sh_order=args.sh_order, basis_type=args.sh_basis)
    visualize_sfs(sfs, sph_gtab, scale=0.5, out_path=args.output)


if __name__ == '__main__':
    main()