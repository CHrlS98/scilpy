#!/usr/bin/env python3
import argparse
import logging
from time import perf_counter
import nibabel as nib
import json
import numpy as np
from scilpy.tractanalysis.grid_intersections import grid_intersections
from scilpy.io.utils import (add_json_args, add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask
from dipy.tracking.streamlinespeed import length
from dipy.io.streamline import (load_tractogram,
                                save_tractogram,
                                StatefulTractogram)
from nibabel.streamlines import detect_format, TrkFile


# endpoint statuses
VALID_ENDPOINT_STATUS = 0
INVALID_DIR_STATUS = 1
INVALID_POS_STATUS = 2


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_mask')

    p.add_argument('out_dict',
                   help='Output voxel to tracks json file.')

    p.add_argument('--min_length', default=0.0, type=float,
                   help='Minimum length in mm.')

    add_verbose_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)
    return p


def validate_dps(parser, sft):
    if 'start_status' not in sft.data_per_streamline:
        parser.error('\'start_status\' not in tractogram dps.')
    if 'end_status' not in sft.data_per_streamline:
        parser.error('\'end_status\' not in tractogram dps.')
    if 'seeds' not in sft.data_per_streamline:
        parser.error('\'seeds\' not in tractogram dps.')


def filter_short_tracks(sft, mask):
    """
    Creer un dictionnaire <voxel id: streamline ids>.
    Pour chaque streamline, on teste si elle est valide.
    Si oui, on prend son voxel id et on l'ajoute dans le
    dictionnaire.
    """
    sft.to_vox()
    sft.to_corner()

    all_crossed_indices = grid_intersections(sft.streamlines)
    vox_strl_map = {}

    t0 = perf_counter()
    logging.info('Filtering tractogram...')
    for strl_id, crossed_indices in enumerate(all_crossed_indices):
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, len(sft.streamlines) - 1))

        # Acceptons tout.
        # Nos clusters vont nous dire si les streamlines sont des outliers.
        voxel_indices = np.unique(crossed_indices.astype(int), axis=0)
        for voxel_id in voxel_indices:
            # mask sure position is inside mask
            if mask[voxel_id[0], voxel_id[1], voxel_id[2]] > 0:
                voxel2str = np.array2string(voxel_id)

                # if the voxel is not in the map yet, create empty list
                if voxel2str not in vox_strl_map:
                    vox_strl_map[voxel2str] = []

                # add strl id to dictionary
                vox_strl_map[voxel2str].append(strl_id)

    logging.info('Filtered tractogram in {:.2f}s'.format(perf_counter() - t0))
    return vox_strl_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram, args.in_mask])
    assert_outputs_exist(parser, args, [args.out_dict])

    tracts_format = detect_format(args.in_tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    t0 = perf_counter()
    logging.info('Loading images...')
    sft = load_tractogram(args.in_tractogram, 'same')
    mask = get_data_as_mask(nib.load(args.in_mask))

    # validate_dps(parser, sft)

    logging.info('Loaded input data in {:.2f}s'.format(perf_counter() - t0))

    vox2tracks_map = filter_short_tracks(sft, mask)

    t0 = perf_counter()
    logging.info('Saving outputs...')

    # output dictionary
    out_json = open(args.out_dict, 'w')
    json.dump(vox2tracks_map, out_json,
              indent=args.indent, sort_keys=args.sort_keys)
    logging.info('Saved outputs in {:.2f}s'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
