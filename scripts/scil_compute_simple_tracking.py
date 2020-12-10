#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple sequential pure-python tracking algorithm.
"""

import argparse
from math import ceil
from itertools import product
import numpy as np
import nibabel as nib
import random
from nibabel.streamlines.tractogram import LazyTractogram

from dipy.tracking import utils as track_utils
from dipy.tracking.streamlinespeed import length
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
from dipy.data import get_sphere
from dipy.direction.peaks import (peak_directions,
                                  reshape_peaks_for_visualization)
from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_sh_basis_args)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.multi_processes import peaks_from_sh


def _build_arg_parser():
    p = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',
                   help='Input file (fODF or peaks).')
    p.add_argument('seed_mask',
                   help='Mask for seeding.')
    p.add_argument('mask',
                   help='Tracking mask.')
    p.add_argument('out_tractogram',
                   help='Output streamlines file.')

    p.add_argument('--input_peaks', default=False, action='store_true',
                   help='Select if the input is a peaks file.')
    p.add_argument('--nupeaks',
                   help='Path to optional nupeaks file.')

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds to use.')

    p.add_argument('--rel_th', default=0.3,
                   help='Relative peaks threshold.')
    p.add_argument('--abs_th', default=0.0,
                   help='Absolute peaks threshold.')
    p.add_argument('--min_peak_angle', default=25,
                   help='Minimum separation angle for peak extraction.')
    p.add_argument('--step_size', default=0.5, type=float,
                   help='Step size in millimeter.')
    p.add_argument('--min_length', type=float, default=10.,
                   help='Minimum length of a streamline in mm.')
    p.add_argument('--max_length', type=float, default=300.,
                   help='Maximum length of a streamline in mm.')
    p.add_argument('--theta', default=60.0, type=float,
                   help='Maximum angle between two steps.')
    p.add_argument('--full_basis', action='store_true',
                   help='Specify the input fODF file is in full SH basis.')

    add_overwrite_arg(p)
    add_sh_basis_args(p)

    return p


def get_peaks_and_nupeaks(args, mask):
    if not args.input_peaks:
        fodf_img = nib.load(args.input)
        fodf = fodf_img.get_fdata(dtype=np.float32)
        sh_order = order_from_ncoef(fodf.shape[-1], args.full_basis)
        sphere = get_sphere('repulsion724')

        peaks, _, _ = peaks_from_sh(
            fodf, sphere, mask,
            relative_peak_threshold=args.rel_th,
            absolute_threshold=args.abs_th,
            min_separation_angle=args.min_peak_angle,
            npeaks=10, sh_basis_type=args.sh_basis,
            is_symmetric=False,
            full_basis=args.full_basis)
    else:
        peaks = nib.load(args.input).get_fdata()

    if args.nupeaks:
        nupeaks = nib.load(args.nupeaks).get_fdata().astype(np.uint8)
    else:
        nupeaks = np.zeros_like(mask, dtype=int)
        nupeaks[mask] = np.count_nonzero(
            np.linalg.norm(peaks, axis=-1),
            axis=-1)[mask]

    return peaks, nupeaks


def get_neighbours_peaks(peaks, curr_pos, curr_dir):
    padded_peaks =\
        np.pad(peaks, ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)), mode='edge')
    neighbors_vox = np.ceil(curr_pos - 1.).astype(int)\
        + np.array(list(product((0, 1), repeat=3))) + 1
    neighbors_vox = neighbors_vox.astype(int)
    neighbors_peaks = peaks[neighbors_vox.T]

    return neighbors_peaks


def get_direction_nn(peaks, nupeaks, curr_pos, curr_dir, max_angle):
    """
    Nearest-neighbor asymmetric direction getter
    """
    get_neighbours_peaks(peaks, curr_pos, curr_dir)
    curr_vox = (curr_pos + .5).astype(int)
    vox_peaks = peaks[tuple(curr_vox)][:nupeaks[tuple(curr_vox)]]
    if len(vox_peaks) == 0:
        # No peaks at voxel
        return None

    # separation plane normal
    n = curr_pos - curr_vox.astype(np.float)
    n /= np.linalg.norm(n)

    if curr_dir is None:
        candidates =\
            vox_peaks[np.dot(vox_peaks, n.reshape(3, 1)).reshape(-1) > 0.]
        # the first direction we take is the maximum candidate peak
        if len(candidates) > 0:
            return candidates[0]
        return None
    else:
        # v is a vector going away from the center of the voxel
        flip_dir = np.dot(n, curr_dir) < 0
        v = -curr_dir if flip_dir else curr_dir
        candidates =\
            vox_peaks[np.dot(vox_peaks, v.reshape(3, 1)).reshape(-1) > 0.]
        if len(candidates) == 0:
            # No valid direction found
            return None

    # return the peak most aligned with direction v
    next_dir = candidates[np.argmax(np.dot(candidates, v))]
    if np.dot(next_dir, v) > np.cos(np.deg2rad(max_angle)):
        return -next_dir if flip_dir else next_dir
    return None


def generate_streamline(seed, peaks, nupeaks, step_size, max_angle):
    # forward pass
    terminate = False
    streamline = []
    streamline.append(seed)
    while not terminate:
        prev_dir = None
        if len(streamline) > 1:
            prev_dir = streamline[-1] - streamline[-2]
            prev_dir /= np.linalg.norm(prev_dir)
        new_dir = get_direction_nn(peaks, nupeaks, streamline[-1],
                                   prev_dir, max_angle)
        if new_dir is None:
            terminate = True
        else:
            new_pos = streamline[-1] + step_size * new_dir
            # Validate new_pos is inside the volume
            new_vox = (new_pos + .5).astype(int)
            terminate =\
                (new_vox < 0).any() or (new_vox + 1 > nupeaks.shape).any()
            if not terminate:
                streamline.append(new_pos)

    if len(streamline) < 2:
        # Case where no direction was taken in forward pass
        # We return because we can't go in opposite direction
        return streamline

    # backward pass
    terminate = False
    while not terminate:
        prev_dir = streamline[0] - streamline[1]
        prev_dir /= np.linalg.norm(prev_dir)
        new_dir = get_direction_nn(peaks, nupeaks,
                                   streamline[0],
                                   prev_dir,
                                   max_angle)
        if new_dir is None:
            terminate = True
        else:
            new_pos = streamline[0] + step_size * new_dir
            # Validate new_pos is inside the volume
            new_vox = (new_pos + .5).astype(int)
            terminate =\
                (new_vox < 0).any() or (new_vox + 1 > nupeaks.shape).any()
            if not terminate:
                streamline = [new_pos] + streamline

    return np.array(streamline)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not args.input_peaks and args.nupeaks:
        parser.error('Invalid argument. Cannot specify '
                     'nupeaks without input peaks')

    inputs = [args.input, args.seed_mask, args.mask]
    output = args.out_tractogram
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    seed_img = nib.load(args.seed_mask)
    mask_img = nib.load(args.mask)

    mask = get_data_as_mask(mask_img, dtype=bool)
    seed_mask = seed_img.get_fdata(dtype=np.float32)

    peaks, nupeaks = get_peaks_and_nupeaks(args, mask)

    voxel_size = seed_img.header.get_zooms()[0]
    voxel_step_size = args.step_size / voxel_size
    vox_min_length = args.min_length / voxel_size
    vox_max_length = args.max_length / voxel_size

    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    # tracking is performed in voxel space
    seeds = track_utils.random_seeds_from_mask(
            seed_mask, np.eye(4), seeds_count=nb_seeds,
            seed_count_per_voxel=seed_per_vox)

    streamlines = []
    for s in seeds:
        streamline = generate_streamline(s, peaks, nupeaks,
                                         voxel_step_size,
                                         args.theta)
        if len(streamline) > 1:
            if vox_min_length <= length(streamline) <= vox_max_length:
                streamlines.append(streamline)

    # Create and save tractogram with streamlines
    tractogram = LazyTractogram(lambda: streamlines,
                                affine_to_rasmm=seed_img.affine)

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(seed_img)
    header = create_tractogram_header(filetype, *reference)

    nib.streamlines.save(tractogram, args.out_tractogram, header=header)


if __name__ == '__main__':
    main()
