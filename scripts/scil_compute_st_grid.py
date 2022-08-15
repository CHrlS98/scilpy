#!/usr/bin/env python3
import argparse
import enum
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
from dipy.io.streamline import (load_tractogram)
from nibabel.streamlines import detect_format, TrkFile


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input short-tracks tractogram.')
    p.add_argument('in_mask', help='Input WM mask.')
    p.add_argument('out_dict', help='Output voxel to tracks json file.')

    p.add_argument('--neighbours_order', type=int, default=0,
                   help='Size of the neighbourhood to consider. [%(default)s]')
    p.add_argument('--all_intersections', action='store_true',
                   help='When set, each short-track is added to all the \n'
                        'voxels it intersects.')

    add_verbose_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)
    return p


def assign_from_grid_intersection(sft, mask):
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


def assign_from_seeds(sft, mask, order):
    seeds = sft.data_per_streamline['seeds']
    vox_strl_map = {}
    logging.info('Filtering tractogram...')
    t0 = perf_counter()

    # generate neighbours offsets list
    nbours_offsets = []
    nbours_range = np.arange(-order, order+1)
    for i in nbours_range:
        for j in nbours_range:
            for k in nbours_range:
                nbours_offsets.append(np.array([i, j, k]))
    nbours_offsets = np.asarray(nbours_offsets)

    # zeropad mask to deal with outside-of-image borders
    if order > 0:
        # trick to keep same indices as in original array
        # negatives will roll to (zero-padded) edge of image.
        mask = np.pad(mask, ((0, order),))

    for strl_id, seed_pos in enumerate(seeds):
        # seed position is in vox space, origin center
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, len(sft.streamlines) - 1))
        vox_ids = (seed_pos + 0.5).astype(int)
        vox_ids = nbours_offsets + vox_ids
        for vox_id in vox_ids:
            if mask[vox_id[0], vox_id[1], vox_id[2]] > 0:
                vox_key = np.array2string(vox_id)
                if vox_key not in vox_strl_map:
                    vox_strl_map[vox_key] = []
                vox_strl_map[vox_key].append(strl_id)
    logging.info('Filtered tractogram in {:.2f}s'.format(perf_counter() - t0))
    return vox_strl_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.in_tractogram, args.in_mask])
    assert_outputs_exist(parser, args, [args.out_dict])

    if detect_format(args.in_tractogram) is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.in_tractogram))

    t0 = perf_counter()
    logging.info('Loading images...')
    sft = load_tractogram(args.in_tractogram, 'same')
    mask = get_data_as_mask(nib.load(args.in_mask))

    logging.info('Loaded input data in {:.2f}s'.format(perf_counter() - t0))

    if args.all_intersections:
        vox2tracks_map = assign_from_grid_intersection(sft, mask)
    else:
        vox2tracks_map = assign_from_seeds(sft, mask, args.neighbours_order)

    t0 = perf_counter()
    logging.info('Saving outputs...')

    # output dictionary
    out_json = open(args.out_dict, 'w')
    json.dump(vox2tracks_map, out_json,
              indent=args.indent, sort_keys=args.sort_keys)
    logging.info('Saved outputs in {:.2f}s'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
