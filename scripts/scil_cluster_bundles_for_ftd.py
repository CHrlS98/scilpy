#!/usr/bin/env python3
import argparse
import logging
import json
from time import perf_counter
import nibabel as nib
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from dipy.io.streamline import save_tractogram, StatefulTractogram
from scilpy.reconst.ftd import ClusterForFTD


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_json',
                   help='Input voxel to tracks dictionary.')
    p.add_argument('out_labels',
                   help='Output label map.')
    p.add_argument('out_centroids',
                   help='Output centroids tractogram.')

    p.add_argument('--max_deviation', type=float, default=40.0,
                   help='Maximum angular deviation between cluster elements.')
    p.add_argument('--batch_size', default=1000,
                   help='Batch size for GPU.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_reference_arg(p)
    return p


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
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    assert_inputs_exist(parser, [args.in_tractogram, args.in_json])
    assert_outputs_exist(parser, args, [args.out_labels, args.out_centroids])

    t0 = perf_counter()
    logging.info('Loading input data...')
    # load tractogram
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    # load json file
    vox2tracks = json.load(open(args.in_json, 'r'))
    logging.info('Loaded input data in {:.2f} seconds'
                 .format(perf_counter() - t0))

    clusters = ClusterForFTD(sft, vox2tracks,
                             max_mean_deviation=args.max_deviation)

    # launch compute
    labels, centroids, centroids_voxel =\
        clusters.cluster_gpu(batch_size=args.batch_size)

    logging.info('Saving outputs...')
    t0 = perf_counter()
    nib.save(nib.Nifti1Image(labels.astype(np.float32), sft.affine),
             args.out_labels)

    # data per streamline is voxel id for each centroid
    dps = {'voxel': np.asarray(centroids_voxel)}

    out_sft = StatefulTractogram.from_sft(centroids, sft,
                                          data_per_streamline=dps)
    save_tractogram(out_sft, args.out_centroids)
    logging.info('Saved outputs in {:.2f}'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
