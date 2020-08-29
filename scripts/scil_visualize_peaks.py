#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize peaks loaded from NIfTI1 image
"""

import argparse

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from fury import window, actor, colormap

from scilpy.io.utils import (add_sh_basis_args)
from scilpy.viz.screenshot import (prepare_texture_slicer_actor,
                                   crop_data_along_axis,
                                   display_scene, create_colormap)

WINDOW_SIZE = (768, 768)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Input peaks image file.')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--peaks_values', help='Values of peaks')

    p.add_argument('--min_value', type=float,
                   help='The minimum value for mapping background colors')

    p.add_argument('--max_value', type=float,
                   help='The maximum value for mapping background colors')

    p.add_argument('--scale', type=float, default=0.5,
                   help='Scaling for peaks amplitude')

    p.add_argument('--line_width', type=float, default=1.0,
                   help='Set line width')

    p.add_argument('--bg',
                   help='Optional background image file')

    p.add_argument('--mask',
                   help='Path to mask')

    p.add_argument('--output',
                   help='Path to output file to write')

    p.add_argument('--silent', default=False, action='store_true',
                   help='Silent mode. No interactive window')

    p.add_argument('--axis_name', default='axial',
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize.')

    p.add_argument('--interactor', default='image',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window')

    p.add_argument('--distinguishable', default=False, action='store_true',
                   help='Use distinguishable color for each integer \
                         value in texture')

    p.add_argument('--color', default=[1.0, 0.0, 0.0], nargs=3, type=float,
                   help='Color ')

    return p


def prepare_peaks_slicer_actor(data, index, values, orientation,
                               mask, line_width, colors):
    peaks_slicer = actor.peak_slicer(data, values, symm=False, mask=mask,
                                     colors=colors,
                                     linewidth=line_width)
    if orientation == 'sagittal':
        peaks_slicer.display_extent(index, index, 0, data.shape[1],
                                    0, data.shape[2])
    elif orientation == 'coronal':
        peaks_slicer.display_extent(0, data.shape[0], index, index,
                                    0, data.shape[2])
    elif orientation == 'axial':
        peaks_slicer.display_extent(0, data.shape[0], 0, data.shape[1],
                                    index, index)
    return peaks_slicer


def get_mask(data, args):
    if args.mask:
        mask = nib.nifti1.load(args.mask).get_fdata().astype(np.bool)
    else:
        mask = np.linalg.norm(data, axis=-1) > 0
        mask = np.sum(mask, axis=-1) > 0
    return mask


def get_peaks_vals(data, args):
    if args.peaks_values:
        peaks_val = nib.nifti1.load(args.peaks_values).get_fdata()

        peaks_max_per_voxel = np.max(peaks_val, axis=-1)
        peaks_val[peaks_max_per_voxel > 0] = np.divide(
            peaks_val[peaks_max_per_voxel > 0],
            peaks_max_per_voxel.reshape(
                np.append(peaks_max_per_voxel.shape, [1]))
            [peaks_max_per_voxel > 0])
    else:
        peaks_val = np.ones(data.shape[:-1])
    return peaks_val * args.scale


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    actors = []
    peaks_data = nib.nifti1.load(args.input).get_fdata()

    mask = get_mask(peaks_data, args)
    peaks_val = get_peaks_vals(peaks_data, args)

    peaks_actor =\
        prepare_peaks_slicer_actor(peaks_data, args.slice_index, peaks_val,
                                   args.axis_name, mask, args.line_width,
                                   args.color)
    # actors.append(peaks_actor)

    if args.bg:
        bg_data = nib.nifti1.load(args.bg).get_fdata()
        bg_data[mask == 0] = 0

        if args.distinguishable:
            colormap_lut = create_colormap(int(bg_data.max() + 1))
        else:
            colormap_lut = None

        bg_actor =\
            prepare_texture_slicer_actor(bg_data, args.min_value,
                                         args.max_value, args.axis_name,
                                         colormap_lut=colormap_lut,
                                         slice_index=args.slice_index)
        actors.append(bg_actor)

    display_scene(actors,
                  peaks_data.shape,
                  WINDOW_SIZE,
                  args.axis_name,
                  args.interactor,
                  args.output,
                  'Visualize peaks',
                  silent=args.silent,
                  slice_index=args.slice_index)


if __name__ == '__main__':
    main()
