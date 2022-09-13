#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to downsample a tractogram per voxel.
Requires dictionary mapping voxels to streamlines.
"""

import argparse
import json
import logging
import numpy as np

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)

from dipy.io.stateful_tractogram import StatefulTractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractography file.')
    p.add_argument('in_vox2tracks',
                   help='Voxel to track id mapping (json).')
    p.add_argument('in_npv', type=int,
                   help='Number of streamlines per voxel in input tractogram.')
    p.add_argument('out_npv', type=int,
                   help='Number per voxel for resampled tractogram.')

    p.add_argument('out_tractogram',
                   help='Output tractography file.')
    p.add_argument('out_vox2tracks',
                   help='Output voxel to track id mapping (json).')

    p.add_argument('--seed', default=None, type=int,
                   help='Use a specific random seed for the resampling.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_vox2tracks])
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not args.out_npv < args.in_npv:
        parser.error('Output npv must be smaller than input.')

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    vox2tracks = json.load(open(args.in_vox2tracks))

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    ratio_keep = float(args.out_npv / args.in_npv)

    strl_keep = []
    vox2tracks_keep = {}
    rng = np.random.RandomState(args.seed)

    strl_ids_offset = 0
    for vox, strl_ids in vox2tracks.items():
        ind = np.arange(len(strl_ids))
        print(len(ind))
        rng.shuffle(ind)
        ind = ind[:int(ratio_keep*len(ind))]
        print(len(ind))
        ids_keep = np.asarray(strl_ids)[ind]
        vox2tracks_keep[vox] = []
        for i in ids_keep:
            strl_keep.append(sft.streamlines[i])
            vox2tracks_keep[vox].append(strl_ids_offset)
            strl_ids_offset += 1

    out_sft = StatefulTractogram.from_sft(strl_keep, sft)
    save_tractogram(out_sft, args.out_tractogram)

    out_json = open(args.out_vox2tracks, 'w')
    json.dump(vox2tracks_keep, out_json,
              indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
