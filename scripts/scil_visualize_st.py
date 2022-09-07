#!/usr/bin/env python3
import argparse
import json
import numpy as np
import nibabel as nib
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, snapshot
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.image import get_data_as_mask

from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from fury import actor, window, colormap


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_st')
    p.add_argument('in_vox2tracks')
    p.add_argument('in_vox2ids')
    p.add_argument('in_mask')
    p.add_argument('in_rois')
    p.add_argument('in_fodf')

    p.add_argument('--opacity', type=float, default=1.0)
    p.add_argument('--output')

    add_reference_arg(p)
    return p


def snapshot_voxel(streamlines, vox2tracks, vox2ids, rois_mask,
                   opacity=None, seeds=None):
    x, y, z = np.nonzero(rois_mask)
    actors = []
    for vox in zip(x, y, z):
        key = np.array2string(np.asarray(vox))
        strl_ids = vox2tracks[key]
        cluster_ids = vox2ids[key]
        nb_clusters = np.max(cluster_ids) + 1
        strl = streamlines[strl_ids]
        colors = colormap.distinguishable_colormap(nb_colors=nb_clusters)

        for cluster_i in range(nb_clusters):
            c_strl = strl[np.asarray(cluster_ids) == cluster_i]
            c_opacity = float(len(c_strl)) / float(len(strl))\
                if opacity is None else opacity
            color = np.tile(colors[cluster_i], len(c_strl)).reshape(-1, 3)
            a = actor.line(c_strl, colors=color, opacity=c_opacity)
            actors.append(a)

        if seeds is not None:
            seeds_a = actor.dots(seeds[strl_ids], color=(1, 1, 1), opacity=0.8)
            actors.append(seeds_a)

    s = window.Scene()
    s.background((1, 1, 1))
    s.add(*actors)
    window.show(s)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vox2tracks = json.load(open(args.in_vox2tracks, 'r'))
    vox2ids = json.load(open(args.in_vox2ids, 'r'))
    rois_mask = get_data_as_mask(nib.load(args.in_rois), dtype=bool)
    crop_mask = get_data_as_mask(nib.load(args.in_mask), dtype=bool)
    rois_mask[np.logical_not(crop_mask)] = False

    sh = nib.load(args.in_fodf).get_fdata(dtype=np.float32)
    sh[np.logical_not(crop_mask)] = 0.0

    sft = load_tractogram_with_reference(parser, args, args.in_st)
    sft.to_vox()
    streamlines = sft.streamlines

    x, y, z = np.nonzero(rois_mask)

    actors = []
    voxels = []
    for vox in zip(x, y, z):
        key = np.array2string(np.asarray(vox))
        strl_ids = vox2tracks[key]
        voxels.append(vox)
        cluster_ids = vox2ids[key]
        nb_clusters = np.max(cluster_ids) + 1
        strl = streamlines[strl_ids]
        colors = colormap.distinguishable_colormap(nb_colors=nb_clusters)

        for cluster_i in range(nb_clusters):
            c_strl = strl[np.asarray(cluster_ids) == cluster_i]
            color = np.tile(colors[cluster_i], len(c_strl)).reshape(-1, 3)
            a = actor.line(c_strl, colors=color, linewidth=2.0,
                           opacity=args.opacity)
            actors.append(a)

    # cast as array
    voxels = np.asarray(voxels, dtype=float)

    cube_a = actor.cube(centers=voxels, colors=(1, 1, 1))
    cube_a.GetProperty().SetOpacity(0.6)
    actors.append(cube_a)
    order, full = get_sh_order_and_fullness(sh.shape[-1])
    sphere = get_sphere('symmetric724')
    B_mat = sh_to_sf_matrix(sphere,
                            sh_order=order,
                            full_basis=full,
                            return_inv=False)

    odf_a = actor.odf_slicer(sh, sphere=sphere, B_matrix=B_mat, opacity=1.0)
    actors.append(odf_a)

    scene = window.Scene()
    scene.add(*actors)
    if args.output:
        # scene.reset_camera()
        snapshot(scene, args.output, size=(500, 500))
    else:
        window.show(scene)


if __name__ == '__main__':
    main()
