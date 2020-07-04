#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize fODFs loaded from NIfTI1 image
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from fury import window, actor

from scilpy.io.utils import (add_sh_basis_args)

WINDOW_SIZE=(768, 768)

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input_fodf',
                   help='Input SH image file.')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--background_image', 
                   help='Optional background image file')

    p.add_argument('--output',
                   help='Path to output file to write')

    p.add_argument('--axis_name', default='axial', 
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize.\
                   One of: [sagittal, coronal, axial]')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH.')

    p.add_argument('--sphere', default='symmetric724',
                   help='Name of the sphere used to reconstruct SF')

    p.add_argument('--radial_scale_off', default=False, 
                   action='store_true', help='Disable radial scale for ODF slicer')

    add_sh_basis_args(p)

    return p


def get_translation_matrix(translation):
    return np.array([[1.0, 0.0, 0.0, translation[0]], 
                     [0.0, 1.0, 0.0, translation[1]],
                     [0.0, 0.0, 1.0, translation[2]],
                     [0.0, 0.0, 0.0, 1.0]])


def prepare_odf_actor(data, sphere, axis_name, radial_scale_off):
    odf_actor = actor.odf_slicer(data,
                                 radial_scale=not(radial_scale_off),
                                 sphere=sphere,
                                 colormap='jet',
                                 scale=0.5)

    if axis_name == 'sagittal':
        odf_actor.display_extent(0, 0, 0, data.shape[1] - 1, 0, data.shape[2] - 1)
    elif axis_name == 'coronal':
        odf_actor.display_extent(0, data.shape[0] - 1, 0, 0, 0, data.shape[2] - 1)
    elif axis_name == 'axial':
        odf_actor.display_extent(0, data.shape[0] - 1, 0, data.shape[1] - 1, 0, 0)

    return odf_actor


def prepare_slicer_actor(data, axis_name):

    if axis_name == 'sagittal':
        slicer_actor = actor.slicer(data, affine=get_translation_matrix((1.0, 0.0, 0.0)),
                                    interpolation='nearest')
        slicer_actor.display_extent(0, 0, 0, data.shape[1] - 1, 0, data.shape[2] - 1)
    elif axis_name == 'coronal':
        slicer_actor = actor.slicer(data, affine=get_translation_matrix((0.0, -1.0, 0.0)),
                                    interpolation='nearest')
        slicer_actor.display_extent(0, data.shape[0] - 1, 0, 0, 0, data.shape[2] - 1)
    elif axis_name == 'axial':
        slicer_actor = actor.slicer(data, affine=get_translation_matrix((0.0, 0.0, 1.0)),
                                    interpolation='nearest')
        slicer_actor.display_extent(0, data.shape[0] - 1, 0, data.shape[1] - 1, 0, 0)

    return slicer_actor


def prepare_scene(axis_name, shape):
    if axis_name == 'sagittal':
        view_position = [-280.0,
                         (shape[1] - 1) / 2.0,
                         (shape[2] - 1) / 2.0]
        view_center = [0.0,
                       (shape[1] - 1) / 2.0,
                       (shape[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
        zoom_factor = 2.0 / shape[1] if shape[1] > shape[2] else 2.0 / shape[2]
    elif axis_name == 'coronal':
        view_position = [(shape[0] - 1) / 2.0,
                         280.0,
                         (shape[2] - 1) / 2.0]
        view_center = [(shape[0] - 1) / 2.0,
                       0.0,
                       (shape[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
        zoom_factor = 2.0 / shape[0] if shape[0] > shape[2] else 2.0 / shape[2]
    elif axis_name == 'axial':
        view_position = [(shape[0] - 1) / 2.0,
                         (shape[1] - 1) / 2.0,
                         -280.0]
        view_center = [(shape[0] - 1) / 2.0,
                         (shape[1] - 1) / 2.0,
                         0.0]
        view_up = [0.0, 1.0, 0.0]
        zoom_factor = 2.0 / shape[0] if shape[0] > shape[1] else 2.0 / shape[1]

    scene = window.Scene()
    scene.projection('parallel')
    scene.set_camera(position=view_position,
                     focal_point=view_center,
                     view_up=view_up)
    scene.zoom(zoom_factor)

    return scene


def display_scene(odf_data, sphere, bg_data, radial_scale_off,
                  orientation, output):
    scene = prepare_scene(orientation, odf_data.shape)

    # Instanciate ODF slicer actor
    odf_actor = prepare_odf_actor(odf_data, sphere, orientation, radial_scale_off)
    scene.add(odf_actor)

    # Prepare error map actor if supplied
    if bg_data is not None:
        bg_actor = prepare_slicer_actor(bg_data, orientation)
        scene.add(bg_actor)

    showm = window.ShowManager(scene, size=WINDOW_SIZE,
                               title='Visualize SH',
                               reset_camera=False,
                               interactor_style='image')
    showm.initialize()
    showm.start()

    if output:
        window.record(scene, size=WINDOW_SIZE, out_path=output, reset_camera=False)


def crop_data_along_axis(data, idx, axis_name):
    if axis_name == 'sagittal':
        return data[idx:idx+1, :, :]
    elif axis_name == 'coronal':
        return data[:, idx:idx+1, :]
    elif axis_name == 'axial':
        return data[:, :, idx:idx+1]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fodf_img = nib.nifti1.load(args.input_fodf)
    fodf_data = fodf_img.get_fdata()

    bg_cropped_data = None
    if args.background_image:
        bg_img = nib.nifti1.load(args.background_image)
        bg_cropped_data =\
             crop_data_along_axis(bg_img.get_fdata(), args.slice_index, args.axis_name)

    fodf_cropped_data =\
        crop_data_along_axis(fodf_data, args.slice_index, args.axis_name)

    sph_gtab = get_sphere(args.sphere)
    odf_data_sf = sh_to_sf(fodf_cropped_data,
                           sph_gtab,
                           sh_order=args.sh_order,
                           basis_type=args.sh_basis)

    display_scene(odf_data_sf, sph_gtab,
                  bg_cropped_data,
                  args.radial_scale_off,
                  args.axis_name,
                  args.output)


if __name__ == '__main__':
    main()