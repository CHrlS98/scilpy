#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from dipy.segment.featurespeed import VectorOfEndpointsFeature
from dipy.segment.metric import CosineMetric
from dipy.segment.clustering import QuickBundlesX
from dipy.tracking.metrics import length
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.ftd import ClusterForFTD


from dipy.io.streamline import load_tractogram
from fury import window, actor
from nibabel.streamlines import detect_format, TrkFile


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_endpoints',
                   help='Interface mask.')
    p.add_argument('out_labels',
                   help='Output label map.')

    add_overwrite_arg(p)
    return p


def _status_to_color(status):
    if status == 0:  # normal
        return [0.0, 1.0, 0.0]  # Green
    if status == 1:  # invalid direction
        return [1.0, 0.0, 0.0]  # Red
    if status == 2:  # invalid position
        return [0.0, 0.0, 1.0]  # Blue


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram, args.in_endpoints])
    assert_outputs_exist(parser, args, args.out_labels)

    tracts_format = detect_format(args.in_tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    # Load files and data. TRKs can have 'same' as reference
    tractogram = load_tractogram(args.in_tractogram, 'same')

    endpoints_img = nib.load(args.in_endpoints)
    endpoint_roi = get_data_as_mask(endpoints_img)

    clusters = ClusterForFTD(tractogram, endpoint_roi)
    clusters.filter_streamlines()
    labels = clusters.cluster_gpu()

    nib.save(nib.Nifti1Image(labels.astype(np.uint8), tractogram.affine),
             args.out_labels)

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


if __name__ == '__main__':
    main()
