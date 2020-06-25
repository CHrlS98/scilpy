#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize asymmetric fODFs loaded from NIfTI1 image
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from fury import window, actor

from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args)

WINDOW_SIZE=(1024, 1024)

def visualize_sfs(sfs, sphere, scale=0.8, out_path=None, window_size=WINDOW_SIZE):
    odfs = actor.odf_slicer(sfs, radial_scale=True,
                            sphere=sphere, colormap='jet', scale=scale)
    dims = sfs.shape
    odfs.display_extent(0, dims[0], 0, dims[1], 0, dims[2])

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

    p.add_argument('input_fodf',
                   help='Input SH image file.')

    p.add_argument('input_error', 
                   help='Input error image file')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--axis_name', default='axial',
                   help='Name of the axis to visualize.\
                   One of: [sagittal, coronal, axial]')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH.')

    p.add_argument('--sphere', default='symmetric724',
                   help='Name of the sphere used to reconstruct SF')

    return p


def get_translation_matrix(translation):
    return np.array([[1.0, 0.0, 0.0, translation[0]], 
                     [0.0, 1.0, 0.0, translation[1]],
                     [0.0, 0.0, 1.0, translation[2]],
                     [0.0, 0.0, 0.0, 1.0]])


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fodf_img = nib.nifti1.load(args.input_fodf)
    fodf_data = fodf_img.get_fdata()

    error_img = nib.nifti1.load(args.input_error)
    error_data = error_img.get_fdata()

    data_slice = None
    if args.axis_name == 'sagittal':
        fodf_data_slice = fodf_data[args.slice_index:args.slice_index + 1, :, :, :]
        error_data_slice = error_data[args.slice_index:args.slice_index + 1]
    elif args.axis_name == 'coronal':
        fodf_data_slice = fodf_data[:, args.slice_index:args.slice_index + 1, :, :]
        error_data_slice = error_data[:, args.slice_index:args.slice_index + 1]
    else:
        fodf_data_slice = fodf_data[:, :, args.slice_index:args.slice_index + 1, :]
        error_data_slice = error_data[:, :, args.slice_index:args.slice_index + 1]

    sph_gtab = get_sphere(args.sphere)

    sfs = sh_to_sf(fodf_data_slice, sph_gtab, sh_order=args.sh_order, basis_type='descoteaux07_full')

    odfs = actor.odf_slicer(sfs, radial_scale=True, sphere=sph_gtab, colormap='jet', scale=0.5)
    odfs.display_extent(0, sfs.shape[0], 0, sfs.shape[1], 0, sfs.shape[2])

    error = actor.slicer(error_data_slice,
                         affine=get_translation_matrix([0.0, 0.0, -1.0]),
                         interpolation='nearest')
    #error.display_extent(0, error_data_slice.shape[0],
    #                     0, error_data_slice.shape[1],
    #                     0, error_data_slice.shape[2])

    scene = window.Scene()
    showm = window.ShowManager(scene, size=WINDOW_SIZE, title='Visualize SH')
    showm.initialize()

    #scene.add(odfs)
    scene.add(error)

    showm.start()

    window.record(scene, size=WINDOW_SIZE, out_path='hcp_reconst_error.png', reset_camera=False)


if __name__ == '__main__':
    main()