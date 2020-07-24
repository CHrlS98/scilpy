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
                                   display_scene)

WINDOW_SIZE = (768, 768)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Input peaks image file.')

    p.add_argument('slice_index', type=int,
                   help='The index of the slice to visualize in axis_name')

    p.add_argument('--min_value', type=float,
                   help='The minimum value for mapping background colors')

    p.add_argument('--max_value', type=float,
                   help='The maximum value for mapping background colors')

    p.add_argument('--background',
                   help='Optional background image file')

    p.add_argument('--output',
                   help='Path to output file to write')

    p.add_argument('--axis_name', default='axial',
                   choices={'axial', 'coronal', 'sagittal'},
                   help='Name of the axis to visualize.')

    p.add_argument('--interactor', default='image',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window')

    p.add_argument('--distinguishable', default=False, action='store_true',
                   help='Use distinguishable color for each integer \
                         value in texture')

    return p


def prepare_peaks_slicer_actor(data, orientation):
    values = np.ones(data.shape[:-1]) * 0.5
    peaks_slicer = actor.peak_slicer(data, values, symm=False)
    if orientation == 'sagittal':
        peaks_slicer.display_extent(0, 0, 0, data.shape[1], 0, data.shape[2])
    elif orientation == 'coronal':
        peaks_slicer.display_extent(0, data.shape[0], 0, 0, 0, data.shape[2])
    elif orientation == 'axial':
        peaks_slicer.display_extent(0, data.shape[0], 0, data.shape[1], 0, 0)
    return peaks_slicer


def create_colormap(nb_colors):
    cm = np.array(colormap.distinguishable_colormap(
            bg=(1.0, 0.0, 0.0),
            exclude=[(0.0, 0.0, 0.0)],
            nb_colors=nb_colors - 1))
    cm = np.vstack(([0, 0, 0], cm))
    lut = colormap.colormap_lookup_table(colors=cm,
                                         scale_range=(0, nb_colors - 1))

    return lut


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    actors = []
    peaks_data = nib.nifti1.load(args.input).get_fdata()
    peaks_cropped_data =\
        crop_data_along_axis(peaks_data, args.slice_index, args.axis_name)
    peaks_actor =\
        prepare_peaks_slicer_actor(peaks_cropped_data, args.axis_name)
    actors.append(peaks_actor)

    bg_cropped_data = None
    if args.background:
        bg_data = nib.nifti1.load(args.background).get_fdata()
        bg_cropped_data =\
            crop_data_along_axis(bg_data, args.slice_index, args.axis_name)

        if args.distinguishable:
            colormap_lut = create_colormap(int(bg_data.max() + 1))
            actors.append(actor.scalar_bar(colormap_lut, nb_labels=0))
        else:
            colormap_lut = None

        bg_actor =\
            prepare_texture_slicer_actor(bg_cropped_data, args.min_value,
                                         args.max_value, args.axis_name,
                                         colormap_lut=colormap_lut)
        actors.append(bg_actor)

    display_scene(actors,
                  peaks_cropped_data.shape,
                  WINDOW_SIZE,
                  args.axis_name,
                  args.interactor,
                  args.output,
                  'Visualize peaks')


if __name__ == '__main__':
    main()
