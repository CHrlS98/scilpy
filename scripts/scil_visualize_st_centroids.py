#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg
from scilpy.viz.scene_utils import (create_scene, create_texture_slicer, render_scene,
                                    create_odf_slicer, create_cmap_lookup)
from dipy.data import get_sphere

from fury import actor


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_centroids')

    # additional images
    p.add_argument('--in_odf')
    p.add_argument('--in_background')
    p.add_argument('--in_mask')

    p.add_argument('--axis', default='coronal',
                   choices=['coronal', 'sagittal', 'axial'],
                   help='Name of anatomical slice to display.')
    p.add_argument('--slice', default=None, type=int,
                   help='Index of slice to display.')
    p.add_argument('--scale', type=float, default=1.0,
                   help='Normalize centroids.')
    p.add_argument('--align_to_grid', action='store_true',
                   help='Align to voxel grid.')
    p.add_argument('--linewidth', default=1.0, type=float,
                   help='Line width.')
    p.add_argument('--in_slice_opacity', type=float, default=1.0,
                   help='In slice opacity.')
    p.add_argument('--out_of_slice_opacity', type=float, default=1.0,
                   help='Out of slice opacity.')
    p.add_argument('--odf_opacity', default=0.8, type=float,
                   help='ODF opacity.')

    add_reference_arg(p)
    return p


def snapshot_slice(streamlines, voxels, shape, orientation,
                   slice_id=None, align_to_grid=False, scale=1.0,
                   out_of_slice_th=0.6, in_slice_opacity=0.6,
                   out_of_slice_opacity=0.2, linewidth=1.0, odf=None,
                   odf_opacity=0.8, background=None):
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

    # all streamlines have the same number of points.
    n_pts = len(streamlines[0])
    mid_pos = (n_pts - 1.0) / 2.0
    before = int(np.floor(mid_pos))
    after = int(np.ceil(mid_pos))
    mid_pts = np.array([[0.5*s[before] + 0.5*s[after]] for s in streamlines])

    streamlines = np.asarray(streamlines) - mid_pts
    streamlines *= scale
    if align_to_grid:
        streamlines += voxel_pos[:, None, :]
    else:
        streamlines += mid_pts

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
        odf_actor = create_odf_slicer(odf, orientation, slice_id, None,
                                      sphere, 1, 8, 'descoteaux07', False,
                                      0.5, True, True, colormap=None,
                                      opacity=odf_opacity)
        actors.append(odf_actor)

    if background is not None:
        _, lut = create_cmap_lookup(0.0, background.max(), 'gray')
        tex_actor = create_texture_slicer(background, orientation, slice_id,
                                          cmap_lut=lut, opacity=0.5,
                                          offset=1.0)
        actors.append(tex_actor)

    in_slice_actor = actor.line(in_slice_tracks, linewidth=linewidth,
                                opacity=in_slice_opacity)
    out_of_slice_actor = actor.line(out_of_slice_tracks, linewidth=linewidth,
                                    opacity=out_of_slice_opacity)
    actors.extend((in_slice_actor, out_of_slice_actor))

    s = create_scene(actors, orientation, slice_id, shape, 'parallel')
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
    if args.in_odf:
        odf = nib.load(args.in_odf).get_fdata()

    background = None
    if args.in_background:
        background = nib.load(args.in_background).get_fdata()

    if args.in_mask:
        mask = get_data_as_mask(nib.load(args.in_mask))
        if args.in_odf:
            odf *= mask[..., None]
        if args.in_background:
            background *= mask

    snapshot_slice(sft.streamlines, voxels, shape, args.axis,
                   args.slice, align_to_grid=args.align_to_grid,
                   scale=args.scale, out_of_slice_th=0.6,
                   in_slice_opacity=args.in_slice_opacity,
                   out_of_slice_opacity=args.out_of_slice_opacity,
                   linewidth=args.linewidth, odf=odf,
                   odf_opacity=args.odf_opacity,
                   background=background)


if __name__ == '__main__':
    main()
