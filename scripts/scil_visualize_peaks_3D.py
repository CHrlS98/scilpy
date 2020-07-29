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
from fury import window, actor, colormap, ui

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

    p.add_argument('--min_value', type=float,
                   help='The minimum value for mapping background colors')

    p.add_argument('--max_value', type=float,
                   help='The maximum value for mapping background colors')

    p.add_argument('--background',
                   help='Optional background image file')

    p.add_argument('--output',
                   help='Path to output file to write')

    p.add_argument('--interactor', default='trackball',
                   choices={'image', 'trackball'},
                   help='Specify interactor mode for vtk window')

    p.add_argument('--distinguishable', default=False, action='store_true',
                   help='Use distinguishable color for each integer \
                         value in texture')

    return p


def prepare_peaks_slicer_actor(data):
    values = np.ones(data.shape[:-1]) * 0.5
    mid_point = (np.array(data.shape) / 2).astype(int)
    peaks_slicer_x = actor.peak_slicer(data, values, symm=False)
    peaks_slicer_y = actor.peak_slicer(data, values, symm=False)
    peaks_slicer_z = actor.peak_slicer(data, values, symm=False)

    peaks_slicer_x.display_extent(mid_point[0], mid_point[0],
                                  0, data.shape[1], 0, data.shape[2])
    peaks_slicer_y.display_extent(0, data.shape[0], mid_point[1], mid_point[1],
                                  0, data.shape[2])
    peaks_slicer_z.display_extent(0, data.shape[0], 0, data.shape[1],
                                  mid_point[2], mid_point[2])

    return [peaks_slicer_x, peaks_slicer_y, peaks_slicer_z]


def prepare_ui_panel(peaks_actor, shape):
    peaks_actor_x = peaks_actor[0]
    peaks_actor_y = peaks_actor[1]
    peaks_actor_z = peaks_actor[2]
    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        peaks_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        peaks_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        peaks_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)

    line_slider_z.on_change = change_slice_z
    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y

    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.background_color = (0, 0, 0)
        label.color = (1, 1, 1)

        return label

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")

    panel = ui.Panel2D(size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")
    panel.center = (170, 120)

    panel.add_element(line_slider_label_x, (0.1, 0.75))
    panel.add_element(line_slider_x, (0.38, 0.75))
    panel.add_element(line_slider_label_y, (0.1, 0.55))
    panel.add_element(line_slider_y, (0.38, 0.55))
    panel.add_element(line_slider_label_z, (0.1, 0.35))
    panel.add_element(line_slider_z, (0.38, 0.35))

    return panel


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
    peaks_actor =\
        prepare_peaks_slicer_actor(peaks_data)
    ui = prepare_ui_panel(peaks_actor, peaks_data.shape)
    actors.append(ui)
    for a in peaks_actor:
        actors.append(a)

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
                  peaks_data.shape,
                  WINDOW_SIZE,
                  'coronal',
                  'custom',
                  args.output,
                  'Visualize peaks')


if __name__ == '__main__':
    main()
