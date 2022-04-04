#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def transform_data_for_pca(streamlines):
    """
    Transform the data for PCA. All streamlines are resampled
    to the maximum number of points and flattened.

    Parameters
    ----------
    streamlines : list of ndarray
        The streamlines.

    Returns
    -------
    data : ndarray (n_streamlines, n_points*3)
        The data for PCA.
    """
    data = np.concatenate(streamlines)
    return data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_corner()
    sft.to_vox()

    strl = sft.streamlines

    line_actor = actor.line(strl)
    s = window.Scene()
    s.add(line_actor)
    window.show(s)


if __name__ == "__main__":
    main()
