#!/usr/bin/env python3
import argparse
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, snapshot
from scilpy.viz.scene_utils import create_scene, render_scene
from PIL import Image

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_centroids')

    p.add_argument('--in_odf')

    p.add_argument('--axis', default='coronal',
                   choices=['coronal', 'sagittal', 'axial'],
                   help='Name of anatomical slice to display.')
    p.add_argument('--slice', default=None, type=int,
                   help='Index of slice to display.')

    add_reference_arg(p)
    return p


def snapshot_slice(streamlines, voxels, shape, orientation,
                   slice_id=None, out_of_slice_th=0.6, odf=None):
    if orientation == 'sagittal':
        axis = 0
    elif orientation == 'coronal':
        axis = 1
    elif orientation == 'axial':
        axis = 2
    else:
        raise ValueError('Invalid value for orientation')

    if slice_id is None:
        slice_id = shape[axis] // 2

    mask = voxels[:, axis] == slice_id
    streamlines = streamlines[mask]
    dirs = np.array([s[-1] - s[0] for s in streamlines]).reshape((-1, 3))
    norms = np.linalg.norm(dirs, axis=-1)
    dirs[norms > 0] /= norms[..., None]

    axis_to_hide = np.zeros((3, 1))
    axis_to_hide[axis] = 1.0
    weigth = np.abs(np.dot(dirs, axis_to_hide)).squeeze()
    in_slice_tracks = streamlines[weigth < out_of_slice_th]
    out_of_slice_tracks = streamlines[weigth >= out_of_slice_th]

    in_slice_actor = actor.line(in_slice_tracks, opacity=0.8)
    out_of_slice_actor = actor.line(out_of_slice_tracks, opacity=0.4)

    s = create_scene([in_slice_actor, out_of_slice_actor],
                     orientation, slice_id, shape)
    s.reset_camera()  # very important!

    render_scene(s, window_size=(1024, 1024), interactor='trackball',
                 output='{}_{}.png'.format(orientation, slice_id), silent=True)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sft = load_tractogram_with_reference(parser, args, args.in_centroids)
    shape = sft.dimensions
    voxels = sft.data_per_streamline['voxel']

    snapshot_slice(sft.streamlines, voxels, shape, args.axis, args.slice)


if __name__ == '__main__':
    main()
