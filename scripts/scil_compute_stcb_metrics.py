#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import nibabel as nib
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_vox2tracks',
                   help='Input voxel-to-track-ids json file.')
    p.add_argument('in_vox2clusters',
                   help='Input voxel-to-cluster-ids json file.')
    p.add_argument('in_reference', help='Reference image.')

    p.add_argument('out_stc_prob',
                   help='Output short-tracks cluster probabilities.')
    p.add_argument('out_nufit',
                   help='Output nufit image.')
    p.add_argument('--rel_th', default=0.1, type=float,
                   help='Relative threshold for nufit image.')

    add_overwrite_arg(p)
    return p


def _key_to_voxel(key):
    voxel = [int(i) for i in key
             .replace('[ ', '').replace('[', '')
             .replace(' ]', '').replace(']', '')
             .replace('  ', ' ')
             .split(' ')]
    return voxel


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_vox2tracks, args.in_vox2clusters])
    assert_outputs_exist(parser, args, [args.out_stc_prob, args.out_nufit])
    if os.path.splitext(args.in_vox2tracks)[1] != '.json':
        parser.error('Invalid extension for file {}'
                     .format(args.in_vox2tracks))
    if os.path.splitext(args.in_vox2clusters)[1] != '.json':
        parser.error('Invalid extension for file {}'
                     .format(args.in_vox2clusters))
    if os.path.splitext(args.in_reference)[1] not in ['.nii', '.gz']:
        parser.error('Invalid extension for file {}'
                     .format(args.in_reference))
    if os.path.splitext(args.out_stc_prob)[1] not in ['.nii', '.gz']:
        parser.error('Invalid extension for file {}'
                     .format(args.out_stc_prob))
    if os.path.splitext(args.out_nufit)[1] not in ['.nii', '.gz']:
        parser.error('Invalid extension for file {}'
                     .format(args.out_nufit))

    # vox2tracks = json.load(args.in_vox2tracks)
    vox2clusters = json.load(open(args.in_vox2clusters, 'r'))
    reference = nib.load(args.in_reference)
    max_nb_clusters = np.max([np.max(vox2clusters[vox_id])
                              for vox_id in vox2clusters.keys()]) + 1
    stc_prob = np.zeros(np.append(reference.shape[:3], max_nb_clusters),
                        dtype=np.float32)
    for vox_id in vox2clusters.keys():
        cluster_ids = np.asarray(vox2clusters[vox_id], dtype=np.uint8)
        bins = np.bincount(cluster_ids)
        vox_id_arr = _key_to_voxel(vox_id)
        stc_prob[tuple(vox_id_arr)][:len(bins)] = np.sort(bins)[::-1]

    count = np.sum(stc_prob, axis=-1)
    stc_prob[count > 0] /= count[count > 0][..., None]
    nib.save(nib.Nifti1Image(stc_prob.astype(np.float32),
                             reference.affine),
             args.out_stc_prob)

    nufit = np.count_nonzero(stc_prob > args.rel_th, axis=-1)
    nib.save(nib.Nifti1Image(nufit.astype(np.uint8), reference.affine),
             args.out_nufit)


if __name__ == '__main__':
    main()
