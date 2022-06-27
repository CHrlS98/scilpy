#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize seeds used to generate the tractogram or bundle.
When tractography was run, each streamline produced by the tracking algorithm
saved its seeding point (its origin).

The tractogram must have been generated from scil_compute_local/pft_tracking.py
with the --save_seeds option.
"""

import argparse
import numpy as np
from dipy.io.streamline import load_tractogram
from fury import window, actor
from nibabel.streamlines import detect_format, TrkFile

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('tractogram',
                   help='Tractogram file (must be trk)')
    p.add_argument('--save',
                   help='If set, save a screenshot of the result in the '
                        'specified filename')
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.tractogram])
    assert_outputs_exist(parser, args, [], [args.save])

    tracts_format = detect_format(args.tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    # Load files and data. TRKs can have 'same' as reference
    tractogram = load_tractogram(args.tractogram, 'same')
    # Streamlines are saved in RASMM but seeds are saved in VOX
    # This might produce weird behavior with non-iso
    tractogram.to_vox()

    streamlines = tractogram.streamlines
    if 'seeds' not in tractogram.data_per_streamline:
        parser.error('Tractogram does not contain seeds')
    seeds = tractogram.data_per_streamline['seeds']
    start_status = tractogram.data_per_streamline['start_status']
    end_status = tractogram.data_per_streamline['end_status']

    print((start_status == end_status).all())

    def _status_to_color(status):
        if status == 0:  # normal
            return [0.0, 1.0, 0.0]
        if status == 1:  # invalid direction
            return [1.0, 0.0, 0.0]
        if status == 2:  # invalid position
            return [0.0, 0.0, 1.0]

    start_pos = np.array([s[0] for s in streamlines])
    end_pos = np.array([s[-1] for s in streamlines])

    start_colors = np.array([_status_to_color(s) for s in start_status])
    end_colors = np.array([_status_to_color(s) for s in end_status])

    # Make display objects
    streamlines_actor = actor.line(streamlines)
    points = actor.dots(seeds, color=(1., 1., 1.))
    start_pts = actor.point(start_pos, colors=start_colors)
    end_pts = actor.point(end_pos, colors=end_colors)

    # Add display objects to canvas
    s = window.Scene()
    s.add(streamlines_actor)
    s.add(points)
    s.add(start_pts)
    s.add(end_pts)

    # Show and record if needed
    if args.save is not None:
        window.record(s, out_path=args.save, size=(1000, 1000))
    window.show(s)


if __name__ == '__main__':
    main()
