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
from scilpy.viz.screenshot import (display_scene,
                                   prepare_texture_slicer_actor,
                                   crop_data_along_axis,
                                   create_colormap)

WINDOW_SIZE = (768, 768)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input_fodf',
                   help='Input SH image file.')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--min_value', type=float,
                   help='The minimum value for mapping background colors')

    p.add_argument('--max_value', type=float,
                   help='The maximum value for mapping background colors')

    p.add_argument('--bg',
                   help='Optional background image file')

    p.add_argument('--mask',
                   help='Optional mask file')

    p.add_argument('--output',
                   help='Path to output file to write')

    p.add_argument('--axis_name', default='axial',
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize.')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH.')

    p.add_argument('--sphere', default='symmetric724',
                   help='Name of the sphere used to reconstruct SF')

    p.add_argument('--scale', default=0.5, type=float,
                   help='Scaling factor for FODF.')

    p.add_argument('--radial_scale_off', default=False,
                   action='store_true',
                   help='Disable radial scale for ODF slicer')

    p.add_argument('--norm_off', default=False,
                   action='store_true',
                   help='Disable normalization of ODF slicer')

    p.add_argument('--silent', default=False, action='store_true',
                   help='Enable silent mode (no interactive window)')

    p.add_argument('--interactor', default='image',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window')

    p.add_argument('--distinguishable', default=False, action='store_true',
                   help='Use distinguishable colormap for background')

    p.add_argument('--offset', type=float, default=0.5,
                   help='Background offset')

    add_sh_basis_args(p)

    return p


def prepare_odf_actor(data, sphere, axis_name, mask,
                      scale, radial_scale_off, norm_off):
    odf_actor = actor.odf_slicer(data,
                                 mask=mask,
                                 radial_scale=not(radial_scale_off),
                                 sphere=sphere,
                                 colormap='jet',
                                 norm=not(norm_off),
                                 scale=scale)

    if axis_name == 'sagittal':
        odf_actor.display_extent(0, 0, 0, data.shape[1] - 1,
                                 0, data.shape[2] - 1)
    elif axis_name == 'coronal':
        odf_actor.display_extent(0, data.shape[0] - 1, 0, 0,
                                 0, data.shape[2] - 1)
    elif axis_name == 'axial':
        odf_actor.display_extent(0, data.shape[0] - 1,
                                 0, data.shape[1] - 1, 0, 0)
    return odf_actor


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sphere = get_sphere(args.sphere).subdivide(1)

    actors = []
    fodf_img = nib.nifti1.load(args.input_fodf)
    fodf_data = fodf_img.get_fdata()
    if args.mask:
        mask = nib.nifti1.load(args.mask).get_fdata().astype(np.bool)
        cropped_mask =\
            crop_data_along_axis(mask, args.slice_index, args.axis_name)
    fodf_cropped_data =\
        crop_data_along_axis(fodf_data, args.slice_index, args.axis_name)
    odf_data_sf = sh_to_sf(fodf_cropped_data,
                           sphere,
                           sh_order=args.sh_order,
                           basis_type=args.sh_basis)
    if not args.mask:
        cropped_mask = np.linalg.norm(odf_data_sf, axis=-1) > 0

    odf_actor =\
        prepare_odf_actor(odf_data_sf, sphere, args.axis_name,
                          cropped_mask, args.scale, args.radial_scale_off,
                          args.norm_off)
    actors.append(odf_actor)

    if args.bg:
        bg_data = nib.nifti1.load(args.bg).get_fdata()
        if args.mask:
            bg_data[np.logical_not(mask)] = 0
        if args.distinguishable:
            colormap_lut = create_colormap(int(bg_data.max() + 1))
        else:
            colormap_lut = None
        bg_cropped_data =\
            crop_data_along_axis(bg_data,
                                 args.slice_index,
                                 args.axis_name)
        bg_actor =\
            prepare_texture_slicer_actor(bg_cropped_data,
                                         args.min_value,
                                         args.max_value,
                                         args.axis_name,
                                         colormap_lut=colormap_lut,
                                         offset=args.offset)
        actors.append(bg_actor)

    display_scene(actors,
                  odf_data_sf.shape,
                  WINDOW_SIZE,
                  args.axis_name,
                  args.interactor,
                  args.output,
                  'Visualize FODF',
                  silent=args.silent)


if __name__ == '__main__':
    main()
