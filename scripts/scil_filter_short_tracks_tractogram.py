#!/usr/bin/env python3
import argparse
import logging
from time import perf_counter
import nibabel as nib
import json
import numpy as np
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
    p.add_argument('out_trk',
                   help='Output cleaned tractogram.')
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


def _filter_short_tracks(sft, interface):
    """
    Creer un dictionnaire <voxel id: streamline ids>.
    Pour chaque streamline, on teste si elle est valide.
    Si oui, on prend son voxel id et on l'ajoute dans le
    dictionnaire.
    """
    sft.to_vox()
    sft.to_corner()

    streamlines = sft.streamlines

    # shift seeds to origin corner
    seeds = sft.data_per_streamline['seeds'] + 0.5
    start_status = sft.data_per_streamline['start_status']
    end_status = sft.data_per_streamline['end_status']

    valid_tracks = np.zeros(len(streamlines), dtype=bool)
    valid_tracks_id = 0  # the id to save in dictionary
    vox_strl_map = {}

    t0 = perf_counter()
    logging.info('Filtering tractogram...')
    for strl_id, strl in enumerate(streamlines):
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, len(streamlines) - 1))
        start_pos, end_pos = strl[0].astype(int), strl[-1].astype(int)
        starts_in_interface = interface[start_pos[0],
                                        start_pos[1],
                                        start_pos[2]]
        ends_in_interface = interface[end_pos[0],
                                      end_pos[1],
                                      end_pos[2]]
        is_valid =\
            (starts_in_interface and
             end_status[strl_id] == VALID_ENDPOINT_STATUS) or\
            (ends_in_interface and
             start_status[strl_id] == VALID_ENDPOINT_STATUS) or\
            (starts_in_interface and ends_in_interface) or\
            (start_status[strl_id] == VALID_ENDPOINT_STATUS and
             end_status[strl_id] == VALID_ENDPOINT_STATUS)

        # update valid tracks mask
        valid_tracks[strl_id] = is_valid

        # add to dictionary if valid
        if is_valid:
            voxel_id = np.array2string(seeds[strl_id].astype(int))
            if voxel_id not in vox_strl_map:
                vox_strl_map[voxel_id] = []
            vox_strl_map[voxel_id].append(valid_tracks_id)
            valid_tracks_id += 1

    logging.info('Filtered tractogram in {:.2f}s'.format(perf_counter() - t0))
    return valid_tracks, vox_strl_map


def filter_short_tracks(sft, min_length=0.0):
    """
    Creer un dictionnaire <voxel id: streamline ids>.
    Pour chaque streamline, on teste si elle est valide.
    Si oui, on prend son voxel id et on l'ajoute dans le
    dictionnaire.
    """
    sft.to_vox()
    sft.to_corner()

    # minimum length in voxel coordinates
    min_len_vox = min_length / sft.voxel_sizes[0]

    streamlines = sft.streamlines

    # shift seeds to origin corner
    seeds = sft.data_per_streamline['seeds'] + 0.5
    start_status = sft.data_per_streamline['start_status']
    end_status = sft.data_per_streamline['end_status']

    valid_tracks = np.zeros(len(streamlines), dtype=bool)
    valid_tracks_id = 0  # the id to save in dictionary
    vox_strl_map = {}

    t0 = perf_counter()
    logging.info('Filtering tractogram...')
    for strl_id, strl in enumerate(streamlines):
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, len(streamlines) - 1))

        # streamline length
        strl_len = length(strl)

        is_valid = (start_status[strl_id] != INVALID_DIR_STATUS and
                    end_status[strl_id] != INVALID_DIR_STATUS and
                    strl_len >= min_len_vox)

        # update valid tracks mask
        valid_tracks[strl_id] = is_valid

        # add to dictionary if valid
        if is_valid:
            voxel_id = np.array2string(seeds[strl_id].astype(int))
            if voxel_id not in vox_strl_map:
                vox_strl_map[voxel_id] = []
            vox_strl_map[voxel_id].append(valid_tracks_id)
            valid_tracks_id += 1

    logging.info('Filtered tractogram in {:.2f}s'.format(perf_counter() - t0))
    return valid_tracks, vox_strl_map


def _filter_short_tracks(sft, interface):
    """
    Creer un dictionnaire <voxel id: streamline ids>.
    Pour chaque streamline, on teste si elle est valide.
    Si oui, on prend son voxel id et on l'ajoute dans le
    dictionnaire.
    """
    sft.to_vox()
    sft.to_corner()

    streamlines = sft.streamlines

    # shift seeds to origin corner
    seeds = sft.data_per_streamline['seeds'] + 0.5
    start_status = sft.data_per_streamline['start_status']
    end_status = sft.data_per_streamline['end_status']

    valid_tracks = np.zeros(len(streamlines), dtype=bool)
    valid_tracks_id = 0  # the id to save in dictionary
    vox_strl_map = {}

    t0 = perf_counter()
    logging.info('Filtering tractogram...')
    for strl_id, strl in enumerate(streamlines):
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, len(streamlines) - 1))
        start_pos, end_pos = strl[0].astype(int), strl[-1].astype(int)
        starts_in_interface = interface[start_pos[0],
                                        start_pos[1],
                                        start_pos[2]]
        ends_in_interface = interface[end_pos[0],
                                      end_pos[1],
                                      end_pos[2]]
        is_valid =\
            (starts_in_interface and
             end_status[strl_id] == VALID_ENDPOINT_STATUS) or\
            (ends_in_interface and
             start_status[strl_id] == VALID_ENDPOINT_STATUS) or\
            (starts_in_interface and ends_in_interface) or\
            (start_status[strl_id] == VALID_ENDPOINT_STATUS and
             end_status[strl_id] == VALID_ENDPOINT_STATUS)

        # update valid tracks mask
        valid_tracks[strl_id] = is_valid

        # add to dictionary if valid
        if is_valid:
            voxel_id = np.array2string(seeds[strl_id].astype(int))
            if voxel_id not in vox_strl_map:
                vox_strl_map[voxel_id] = []
            vox_strl_map[voxel_id].append(valid_tracks_id)
            valid_tracks_id += 1

    logging.info('Filtered tractogram in {:.2f}s'.format(perf_counter() - t0))
    return valid_tracks, vox_strl_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exist(parser, args, [args.out_trk, args.out_dict])

    tracts_format = detect_format(args.in_tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    t0 = perf_counter()
    logging.info('Loading images...')
    sft = load_tractogram(args.in_tractogram, 'same')
    validate_dps(parser, sft)

    logging.info('Loaded input data in {:.2f}s'.format(perf_counter() - t0))

    valid_tracks, vox2tracks_map = filter_short_tracks(sft, args.min_length)

    t0 = perf_counter()
    logging.info('Saving outputs...')
    # output tractogram
    out_sft = StatefulTractogram.from_sft(sft.streamlines[valid_tracks], sft)
    save_tractogram(out_sft, args.out_trk)

    # output dictionary
    out_json = open(args.out_dict, 'w')
    json.dump(vox2tracks_map, out_json,
              indent=args.indent, sort_keys=args.sort_keys)
    logging.info('Saved outputs in {:.2f}s'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
