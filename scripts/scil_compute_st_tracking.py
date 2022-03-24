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

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _gen_label(vox_id, vshape, sph_id):
    return sph_id * np.prod(vshape) +\
        vox_id[2] * np.prod(vshape[:2]) +\
        vox_id[1] * vshape[0] +\
        vox_id[0]


def _label_streamline(start_dir, start_pos, vshape, sphere):
    start_dir /= np.linalg.norm(start_dir)
    strl_vox = start_pos.astype(int)
    sdir_bin = sphere.find_closest(start_dir)
    return _gen_label(strl_vox, vshape, sdir_bin)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    vshape = sft.dimensions

    labels = sft.data_per_streamline['labels']
    reverse_labels = sft.data_per_streamline['reverse_labels']
    union_labels = np.append(labels, reverse_labels)

    union_labels = union_labels[np.logical_not(np.isnan(union_labels))]\
        .astype(int)
    print(np.min(union_labels), np.max(union_labels))

    hist = np.bincount(union_labels)
    print(np.argmax(hist), np.max(hist))

    idx_mask = np.logical_or(labels == np.argmax(hist),
                             reverse_labels == np.argmax(hist))

    strl = sft.streamlines[idx_mask.squeeze()]
    line_actor = actor.line(strl)
    s = window.Scene()
    s.add(line_actor)
    window.show(s)


if __name__ == "__main__":
    main()
