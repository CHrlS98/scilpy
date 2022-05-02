#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete streamlines from short-tracks tractogram.
"""
import argparse
import numpy as np
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_reference_arg, add_overwrite_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.gpuparallel.octree import OcTree


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('in_seed', help='Input seed file (.nii.gz).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    p.add_argument('--search_radius', type=float, default=0.5,
                   help='Search radius in mm. [%(default)s]')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_seed])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()
    all_strl = np.concatenate(sft.streamlines, axis=0)
    vox_radius = args.search_radius / sft.voxel_sizes[0]
    print("sft.voxel_size: {}".format(sft.voxel_sizes))
    print("Voxel radius:", vox_radius)
    strl_int = (all_strl/vox_radius).astype(int)
    dims = np.max(strl_int, axis=0) + 1

    flat_indices = np.ravel_multi_index(strl_int.T, dims)
    unique, unique_counts = np.unique(flat_indices, return_counts=True)
    print("Number of points:", len(all_strl))
    print("Number of bins:", len(unique))
    print("Max density:", np.max(unique_counts))


if __name__ == '__main__':
    main()
