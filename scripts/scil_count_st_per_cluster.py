#!/usr/bin/env python3
"""
NOTE: Clusters with only one streamline are included in the image.
"""
import argparse
import os
import json
import numpy as np
import nibabel as nib
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.ftd import key_to_vox_index


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_vox2clusters',
                   help='Input voxel-to-cluster-ids json file.')
    p.add_argument('in_reference',
                   help='Reference image.')

    p.add_argument('out_nb_tracks',
                   help='Output short-tracks count per cluster.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_vox2clusters, args.in_reference])
    assert_outputs_exist(parser, args, [args.out_nb_tracks])
    if os.path.splitext(args.in_vox2clusters)[1] != '.json':
        parser.error('Invalid extension for file {}'
                     .format(args.in_vox2clusters))
    if os.path.splitext(args.in_reference)[1] not in ['.nii', '.gz']:
        parser.error('Invalid extension for file {}'
                     .format(args.in_reference))
    if os.path.splitext(args.out_nb_tracks)[1] not in ['.nii', '.gz']:
        parser.error('Invalid extension for file {}'
                     .format(args.out_nb_tracks))

    vox2clusters = json.load(open(args.in_vox2clusters, 'r'))
    reference = nib.load(args.in_reference)
    volume_shape = reference.shape[:3]

    max_nb_clusters = np.max([np.max(vox2clusters[vox_id])
                              for vox_id in vox2clusters.keys()]) + 1
    nb_tracks_per_cluster = np.zeros(np.append(volume_shape, max_nb_clusters),
                                     dtype=np.float32)

    for vox_id in vox2clusters.keys():
        cluster_ids = np.asarray(vox2clusters[vox_id], dtype=int)
        vox_id_arr = key_to_vox_index(vox_id)

        # number of elements in each cluster
        bins = np.bincount(cluster_ids)
        nb_tracks_per_cluster[tuple(vox_id_arr)][:len(bins)] =\
            np.sort(bins)[::-1]

    nib.save(nib.Nifti1Image(nb_tracks_per_cluster.astype(np.float32),
                             reference.affine),
             args.out_nb_tracks)


if __name__ == '__main__':
    main()
