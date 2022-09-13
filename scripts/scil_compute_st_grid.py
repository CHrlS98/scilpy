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

    add_verbose_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)
    return p


def assign_from_seeds(lazy_seeds, nb_streamlines, mask, order):
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

    for strl_id, seed_pos in enumerate(lazy_seeds):
        # seed position is in vox space, origin center
        if(strl_id % 50000 == 0):
            logging.info('Streamline {}/{}'
                         .format(strl_id, nb_streamlines))
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
    mask = get_data_as_mask(nib.load(args.in_mask))

    lazy_trk = nib.streamlines.load(args.in_tractogram, True)
    nb_streamlines = lazy_trk.header['nb_streamlines']
    lazy_seeds = lazy_trk.tractogram.data_per_streamline['seeds']
    vox2tracks_map = assign_from_seeds(lazy_seeds, nb_streamlines,
                                       mask, args.neighbours_order)

    t0 = perf_counter()
    logging.info('Saving outputs...')

    # output dictionary
    out_json = open(args.out_dict, 'w')
    json.dump(vox2tracks_map, out_json,
              indent=args.indent, sort_keys=args.sort_keys)
    logging.info('Saved outputs in {:.2f}s'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
