#!/usr/bin/env python3

import argparse
import nibabel as nib
import numpy as np

from scilpy.viz.scene_utils import create_peaks_slicer, create_scene, render_scene

RGB_COLORS = [(1.0, 1.0, 1.0),
              (0.0, 0.7, 0.0),
              (0.0, 0.0, 0.7)]


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in_peaks', nargs='+', required=True)
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
    for it, fname in enumerate(args.in_peaks):
        vol = nib.load(fname).get_fdata()
        if vol.shape[-1] != 3:
            vol = vol.reshape(np.append(vol.shape[:3], (-1, 3)))
            if slice_index is not None:
                slice_index = vol.shape[1] // 2
            if volume_shape is not None:
                assert volume_shape == vol.shape[:3]
            else:
                volume_shape = vol.shape[:3]
        peak_values = np.full(vol.shape[:-1], 0.5)
        actors.append(create_peaks_slicer(vol, args.axis_name, slice_index, opacity=0.8,
                                          symmetric=False, color=RGB_COLORS[it],
                                          peak_values=peak_values))

    scene = create_scene(actors, args.axis_name, slice_index, volume_shape)
    render_scene(scene, (724, 724), 'trackball', None, False)


if __name__ == '__main__':
    main()
