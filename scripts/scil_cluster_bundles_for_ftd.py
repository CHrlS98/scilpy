#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.ftd import ClusterForFTD

from dipy.io.streamline import (load_tractogram,
                                save_tractogram,
                                StatefulTractogram)
from fury import window, actor
from nibabel.streamlines import detect_format, TrkFile


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_interface',
                   help='Interface mask.')
    p.add_argument('out_labels',
                   help='Output label map.')

    p.add_argument('--out_cleaned_trk',
                   help='Output cleaned tractogram.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def _status_to_color(status):
    if status == 0:  # normal
        return [0.0, 1.0, 0.0]  # Green
    if status == 1:  # invalid direction
        return [1.0, 0.0, 0.0]  # Red
    if status == 2:  # invalid position
        return [0.0, 0.0, 1.0]  # Blue


def show(clusters):
    mask = clusters.all_valid_mask
    streamlines = clusters.streamlines[mask]
    seeds = clusters.seeds[mask]
    start_status = clusters.start_status[mask]
    end_status = clusters.end_status[mask]

    start_pos = np.array([s[0] for s in streamlines])
    end_pos = np.array([s[-1] for s in streamlines])
    start_colors = np.array([_status_to_color(s) for s in start_status])
    end_colors = np.array([_status_to_color(s) for s in end_status])

    # Make display objects
    streamlines_actor = actor.line(streamlines)
    points = actor.dots(seeds, color=(1., 1., 1.))
    start_pts = actor.point(start_pos, colors=start_colors,
                            phi=4, theta=4)

    # Add display objects to canvas
    s = window.Scene()
    s.add(streamlines_actor)
    s.add(points)
    # s.add(start_pts)

    # Show
    window.show(s)


def validate_dps(parser, sft):
    if 'start_status' not in sft.data_per_streamline:
        parser.error('\'start_status\' not in tractogram dps.')
    if 'end_status' not in sft.data_per_streamline:
        parser.error('\'end_status\' not in tractogram dps.')
    if 'seeds' not in sft.data_per_streamline:
        parser.error('\'seeds\' not in tractogram dps.')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram, args.in_interface])
    assert_outputs_exist(parser, args, args.out_labels, args.out_cleaned_trk)

    tracts_format = detect_format(args.in_tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    sft = load_tractogram(args.in_tractogram, 'same')
    validate_dps(parser, sft)

    interface_img = nib.load(args.in_interface)
    interface = get_data_as_mask(interface_img)

    clusters = ClusterForFTD(sft, interface)
    clusters.filter_streamlines()

    labels = clusters.cluster_gpu()

    if args.out_cleaned_trk:
        out_sft = StatefulTractogram.from_sft(
            clusters.streamlines[clusters.all_valid_mask],
            sft)
        save_tractogram(out_sft, args.out_cleaned_trk)

    nib.save(nib.Nifti1Image(labels.astype(np.float32), sft.affine),
             args.out_labels)

    # show(clusters)


if __name__ == '__main__':
    main()
