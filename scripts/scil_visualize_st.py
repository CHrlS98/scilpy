#!/usr/bin/env python3
import argparse
import json
import numpy as np
import nibabel as nib
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg, snapshot,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.image import get_data_as_mask

from dipy.tracking.streamlinespeed import set_number_of_points
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
    p.add_argument('--max_per_cluster', type=int, default=5)
    p.add_argument('--output')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def compute_centroid(bundle):
    bundle = np.asarray([set_number_of_points(s, 20) for s in bundle])
    centroids = bundle[0]
    n_lines = 1
    for i, line in enumerate(bundle[1:]):
        dist_strt = np.mean(np.sum((centroids/n_lines - line)**2.0,
                            axis=-1))
        dist_flip = np.mean(np.sum((centroids/n_lines - line[::-1])**2.0,
                            axis=-1))
        if dist_flip < dist_strt:
            centroids += line[::-1]
        else:
            centroids += line
        n_lines += 1
    return centroids / n_lines


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, required=[args.in_st, args.in_vox2tracks,
                                          args.in_vox2ids, args.in_mask,
                                          args.in_mask, args.in_rois,
                                          args.in_fodf])
    assert_outputs_exist(parser, args, [], optional=args.output)

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
        outlier_gray = (0.5, 0.5, 0.5)
        colors = colormap.distinguishable_colormap(nb_colors=nb_clusters,
                                                   exclude=[outlier_gray])

        for cluster_i in range(nb_clusters):
            c_strl = strl[np.asarray(cluster_ids) == cluster_i]
            if len(c_strl) < 2:
                # clusters containing only one member are outliers!
                centroid = c_strl
                color = np.asarray([outlier_gray])
            else:
                centroid = compute_centroid(c_strl)[None, ...]
                color = colors[cluster_i].reshape(-1, 3)
                txt_a = actor.text_3d('{}'.format(len(c_strl)),
                                      font_size=0.6, shadow=False,
                                      position=centroid[0, 0] +
                                      np.array([0.0, 0.0, 1.0]),
                                      color=color)
                actors.append(txt_a)
            # draw centroids
            a = actor.line(centroid, colors=color.reshape(-1, 3),
                           linewidth=6.0, opacity=args.opacity)
            actors.append(a)

    # cast as array
    voxels = np.asarray(voxels, dtype=float)

    cube_a = actor.cube(centers=voxels, colors=(1, 1, 1))
    cube_a.GetProperty().SetOpacity(0.8)
    actors.append(cube_a)
    order, full = get_sh_order_and_fullness(sh.shape[-1])
    sphere = get_sphere('symmetric724')
    B_mat = sh_to_sf_matrix(sphere, sh_order=order,
                            full_basis=full,
                            return_inv=False)

    odf_a = actor.odf_slicer(sh, sphere=sphere, B_matrix=B_mat, opacity=0.6)
    actors.append(odf_a)

    scene = window.Scene()
    scene.add(*actors)
    if args.output:
        # scene.reset_camera()
        snapshot(scene, args.output, size=(1000, 1000))
    else:
        window.show(scene)


if __name__ == '__main__':
    main()
