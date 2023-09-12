#!/usr/bin/env python3

import argparse
import nibabel as nib
import numpy as np

from scilpy.viz.scene_utils import (create_peaks_slicer,
                                    create_scene, render_scene)

RGB_COLORS = [(1.0, 1.0, 1.0),
              (0.0, 0.7, 0.0),
              (0.0, 0.0, 0.7)]


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_peaks')
    p.add_argument('--in_peaks_values')
    p.add_argument('--slice_index', default=None, type=int)
    p.add_argument('--axis_name', default='coronal',
                   choices=['sagittal', 'coronal', 'axial'])
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    actors = []
    slice_index = args.slice_index
    volume_shape = None

    vol = nib.load(args.in_peaks).get_fdata()
    if vol.shape[-1] != 3:
        vol = vol.reshape(np.append(vol.shape[:3], (-1, 3)))
        if slice_index is not None:
            slice_index = vol.shape[1] // 2
        if volume_shape is not None:
            assert volume_shape == vol.shape[:3]
        else:
            volume_shape = vol.shape[:3]

    if args.in_peaks_values:
        peak_values = nib.load(args.in_peaks_values).get_fdata()
        peak_values /= np.max(peak_values)
    else:
        peak_values = np.full(vol.shape[:-1], 0.5)

    actors.append(create_peaks_slicer(vol, args.axis_name, slice_index,
                                      symmetric=False, color=None,
                                      peak_values=peak_values))

    scene = create_scene(actors, args.axis_name, slice_index, volume_shape)
    render_scene(scene, (724, 724), 'trackball', None, False)


if __name__ == '__main__':
    main()
