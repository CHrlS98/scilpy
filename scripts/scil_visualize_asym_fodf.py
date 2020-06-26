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

WINDOW_SIZE=(768, 768)

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input_fodf',
                   help='Input SH image file.')

    p.add_argument('input_error', 
                   help='Input error image file')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--output',
                   help='Path to output file to write')

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


def display_scene(odf_data, sphere, error_data, shape3D,
                  orientation, slice_index,output):
    odf_actor = actor.odf_slicer(odf_data,
                                 radial_scale=True,
                                 sphere=sphere,
                                 colormap='jet',
                                 scale=0.5)
    if orientation == 'sagittal':
        affine_transform = get_translation_matrix([1.0 - slice_index, 0.0, 0.0])
    elif orientation == 'coronal':
        affine_transform = get_translation_matrix([0.0, -1.0 - slice_index, 0.0])
    elif orientation == 'axial':
        affine_transform = get_translation_matrix([0.0, 0.0, -1.0 - slice_index])
    else:
        print('Invalid orientation')
        return -1

    error_actor = actor.slicer(error_data,
                               affine=affine_transform,
                               interpolation='nearest')

    scene = window.Scene()
    scene.projection('parallel')

    scene.add(odf_actor)
    scene.add(error_actor)
    if orientation == 'sagittal':
        error_actor.display_extent(slice_index, slice_index, 0, shape3D[1] - 1, 0, shape3D[2] - 1)
        odf_actor.display_extent(0, 0, 0, shape3D[1] - 1, 0, shape3D[2] - 1)
        view_position = [-280.0,
                         (shape3D[1] - 1) / 2.0,
                         (shape3D[2] - 1) / 2.0]
        view_center = [0.0,
                       (shape3D[1] - 1) / 2.0,
                       (shape3D[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
    elif orientation == 'coronal':
        error_actor.display_extent(0, shape3D[0] - 1, slice_index, slice_index, 0, shape3D[2] - 1)
        odf_actor.display_extent(0, shape3D[0] - 1, 0, 0, 0, shape3D[2] - 1)
        view_position = [(shape3D[0] - 1) / 2.0,
                         280.0,
                         (shape3D[2] - 1) / 2.0]
        view_center = [(shape3D[0] - 1) / 2.0,
                       0.0,
                       (shape3D[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
    elif orientation == 'axial':
        error_actor.display_extent(0, shape3D[0] - 1, 0, shape3D[1] - 1, slice_index, slice_index)
        odf_actor.display_extent(0, shape3D[0] - 1, 0, shape3D[1] - 1, 0, 0)
        view_position = [(shape3D[0] - 1) / 2.0,
                         (shape3D[1] - 1) / 2.0,
                         280.0]
        view_center = [(shape3D[0] - 1) / 2.0,
                         (shape3D[1] - 1) / 2.0,
                         0.0]
        view_up = [0.0, 1.0, 0.0]
    else:
        print('Invalid orientation')
        return -1

    scene.set_camera(position=view_position,
                     focal_point=view_center,
                     view_up=view_up)
    showm = window.ShowManager(scene, size=WINDOW_SIZE,
                               title='Visualize SH',
                               reset_camera=False,
                               interactor_style='image')
    showm.initialize()
    showm.start()

    if output:
        window.record(scene, size=WINDOW_SIZE, out_path=output, reset_camera=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fodf_img = nib.nifti1.load(args.input_fodf)
    fodf_data = fodf_img.get_fdata()

    error_img = nib.nifti1.load(args.input_error)
    error_data = error_img.get_fdata()

    idx = args.slice_index
    if args.axis_name == 'sagittal':
        fodf_cropped_data = fodf_data[idx:idx+1, :, :, :]
    elif args.axis_name == 'coronal':
        fodf_cropped_data = fodf_data[:, idx:idx+1, :, :]
    elif args.axis_name == 'axial':
        fodf_cropped_data = fodf_data[:, :, idx:idx+1, :]
    else:
        print('Invalid axis name')
        return -1

    sph_gtab = get_sphere(args.sphere)

    odf_data_sf = sh_to_sf(fodf_cropped_data,
                                 sph_gtab,
                                 sh_order=args.sh_order,
                                 basis_type='descoteaux07_full')

    shape = error_data.shape[:3]
    display_scene(odf_data_sf, sph_gtab,
                  error_data, error_data.shape[:3],
                  args.axis_name, args.slice_index,
                  args.output)


if __name__ == '__main__':
    main()