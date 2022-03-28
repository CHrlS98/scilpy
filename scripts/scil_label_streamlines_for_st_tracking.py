#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import argparse
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist)

from dipy.data import get_sphere
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _gen_label(vox_id, sph_id, vshape):
    multi_index = tuple(vox_id) + (sph_id,)
    ravel_index = np.ravel_multi_index(multi_index, vshape)
    return ravel_index


def _label_streamline(start_dir, start_pos, vshape, sphere):
    start_dir /= np.linalg.norm(start_dir)
    strl_vox = start_pos.astype(int)
    sdir_bin = sphere.find_closest(start_dir)
    return _gen_label(strl_vox, sdir_bin,
                      tuple(vshape) + (len(sphere.vertices),))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    vshape = sft.dimensions

    sft.to_vox()
    sft.to_corner()

    sphere = get_sphere('symmetric362')

    streamlines = sft.streamlines
    labels_per_strl = np.full((len(streamlines), 1), np.NaN)
    reverse_strl_label = np.full((len(streamlines), 1), np.NaN)
    for idx, s in enumerate(streamlines):
        if len(s) < 2:  # skip degenerate streamlines
            continue

        labels_per_strl[idx] =\
            _label_streamline(s[1] - s[0], s[0], vshape, sphere)
        reverse_strl_label[idx] =\
            _label_streamline(s[-2] - s[-1], s[-1], vshape, sphere)

    new_dps = sft.data_per_streamline
    new_dps['labels'] = labels_per_strl
    new_dps['reverse_labels'] = reverse_strl_label
    out_sft = StatefulTractogram.from_sft(streamlines, sft,
                                          data_per_streamline=new_dps)
    save_tractogram(out_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
