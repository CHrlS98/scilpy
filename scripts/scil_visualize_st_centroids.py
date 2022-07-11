#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg
from scilpy.viz.scene_utils import (create_scene, render_scene,
                                    create_odf_slicer)
from dipy.data import get_sphere
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
    p.add_argument('--normalize', action='store_true',
                   help='Normalize centroids.')

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
    voxel_pos = voxels[mask]

    mid_pts = np.array([[s[len(s)//2]] for s in streamlines])

    # center around mid pts
    streamlines = np.asarray(streamlines) - mid_pts

    # scale
    streamlines /= 10.0

    # relocate
    streamlines += voxel_pos[:, None, :]

    dirs = np.array([s[-1] - s[0] for s in streamlines]).reshape((-1, 3))
    norms = np.linalg.norm(dirs, axis=-1)
    dirs[norms > 0] /= norms[..., None]

    axis_to_hide = np.zeros((3, 1))
    axis_to_hide[axis] = 1.0
    weigth = np.abs(np.dot(dirs, axis_to_hide)).squeeze()
    in_slice_tracks = streamlines[weigth < out_of_slice_th]
    out_of_slice_tracks = streamlines[weigth >= out_of_slice_th]

    sphere = get_sphere('symmetric724')
    actors = []
    if odf is not None:
        odf_actor = create_odf_slicer(odf, orientation, slice_id, None, sphere,
                                      0, 8, 'descoteaux07', False, 0.5, True,
                                      False, colormap=None, opacity=0.8)
        actors.append(odf_actor)

    in_slice_actor = actor.streamtube(in_slice_tracks, linewidth=0.1, opacity=0.6)
    out_of_slice_actor = actor.streamtube(out_of_slice_tracks, linewidth=0.1, opacity=0.2)
    actors.extend((in_slice_actor, out_of_slice_actor))

    s = create_scene(actors, orientation, slice_id, shape, 'perpective')
    s.reset_camera()  # very important!

    render_scene(s, window_size=(4096, 4096), interactor='trackball',
                 output='{}_{}.png'.format(orientation, slice_id),
                 silent=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sft = load_tractogram_with_reference(parser, args, args.in_centroids)
    shape = sft.dimensions
    voxels = sft.data_per_streamline['voxel']
    sft.to_vox()  # convert to vox space

    odf = None
    if args.in_odf is not None:
        odf = nib.load(args.in_odf).get_fdata()

    snapshot_slice(sft.streamlines, voxels, shape,
                   args.axis, args.slice, 0.6, odf)


if __name__ == '__main__':
    main()
