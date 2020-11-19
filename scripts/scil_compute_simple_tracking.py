#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple sequential pure-python tracking algorithm.
"""

import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines.tractogram import LazyTractogram

from dipy.tracking import utils as track_utils
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
    p.add_argument('in_fodf',
                   help='Input fODF file.')
    p.add_argument('seed_mask',
                   help='Mask for seeding.')
    p.add_argument('mask',
                   help='Tracking mask.')
    p.add_argument('out_tractogram',
                   help='Output streamlines file.')
    p.add_argument('--full_basis', action='store_true',
                   help='Specify the input fODF file is in full SH basis.')
    p.add_argument('--npv', default=5, type=int,
                   help='Number of seeds per voxel.')
    p.add_argument('--rel_th', default=0.3,
                   help='Relative peaks threshold.')
    p.add_argument('--abs_th', default=0.0,
                   help='Absolute peaks threshold.')
    p.add_argument('--min_peak_angle', default=25,
                   help='Minimum separation angle for peak extraction.')
    p.add_argument('--step_size', default=0.5, type=float,
                   help='Step size in millimeter.')
    p.add_argument('--theta', default=60.0, type=float,
                   help='Maximum angle between two steps.')
    p.add_argument('--rand_seed', type=int,
                   help='Random seed used in seed generator.')
    add_overwrite_arg(p)
    add_sh_basis_args(p)

    return p


def get_direction_nn(peaks, nupeaks, curr_pos, curr_dir, max_angle):
    curr_vox = (curr_pos + .5).astype(int)
    if (np.count_nonzero(curr_vox < 0) > 0 or
            np.count_nonzero(curr_vox + 1 > nupeaks.shape) > 0):
        # we are outside the volume
        return None

    vox_peaks = peaks[tuple(curr_vox)][:nupeaks[tuple(curr_vox)]]
    if len(vox_peaks) == 0:
        # No peaks at voxel
        return None

    # separation plane normal
    n = curr_pos - curr_vox.astype(np.float)
    n /= np.linalg.norm(n)

    # curr_dir can be None
    if curr_dir is None:
        candidates = vox_peaks[np.dot(vox_peaks, n.reshape(3, 1)).reshape(-1) > 0.]
        # the first direction we take is the maximum candidate peak
        if len(candidates) > 0:
            return candidates[0]
        else:
            return None
    else:
        # v is a vector going away from the center of the voxel
        v = -curr_dir if np.dot(n, curr_dir) < 0 else curr_dir
        candidates = vox_peaks[np.dot(vox_peaks, v.reshape(3, 1)).reshape(-1) > 0.]
        if len(candidates) == 0:
            # No valid direction found
            return None

    # return the peak most aligned with direction v
    next_dir = candidates[np.argmax(np.dot(candidates, v))]
    if np.dot(next_dir, v) > np.cos(np.deg2rad(max_angle)):
        return next_dir
    return None


def generate_streamline(seed, peaks, nupeaks, step_size, max_angle):
    terminate = False

    # forward pass
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
            if len(streamline) > 1:
                prev_dir = streamline[-1] - streamline[-2]
                prev_dir /= np.linalg.norm(prev_dir)
                if np.dot(prev_dir, new_dir) < 0.:
                    new_dir *= -1.
            new_pos = streamline[-1] + step_size * new_dir
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
            if len(streamline) > 1:
                prev_dir = streamline[0] - streamline[1]
                prev_dir /= np.linalg.norm(prev_dir)
                if np.dot(prev_dir, new_dir) < 0.:
                    new_dir *= -1.
            new_pos = streamline[0] + step_size * new_dir
            streamline = [new_pos] + streamline

    return streamline


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    inputs = [args.in_fodf, args.seed_mask, args.mask]
    output = args.out_tractogram
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, output)

    fodf_img = nib.load(args.in_fodf)
    seed_img = nib.load(args.seed_mask)
    mask_img = nib.load(args.mask)

    mask = get_data_as_mask(mask_img, dtype=bool)
    seed_mask = seed_img.get_fdata(dtype=np.float32)

    # Let's track in voxel space, then transform
    # our tracts using the image affine.
    seeds = track_utils.random_seeds_from_mask(
        seed_mask, np.eye(4),
        seeds_count=args.npv,
        random_seed=args.rand_seed)

    fodf = fodf_img.get_fdata(dtype=np.float32)
    sh_order = order_from_ncoef(fodf.shape[-1], args.full_basis)
    sphere = get_sphere('repulsion724')

    peaks, _, indices = peaks_from_sh(
        fodf, sphere, mask,
        relative_peak_threshold=args.rel_th,
        absolute_threshold=args.abs_th,
        min_separation_angle=args.min_peak_angle,
        npeaks=10, sh_basis_type=args.sh_basis,
        is_symmetric=False,
        full_basis=args.full_basis)
    nupeaks = np.zeros_like(mask, dtype=int)
    nupeaks[mask] = np.sum(indices.astype(int) >= 0, axis=-1)[mask]

    voxel_size = fodf_img.header.get_zooms()[0]
    voxel_step_size = args.step_size / voxel_size
    streamlines = []
    for s in seeds:
        streamline = generate_streamline(s, peaks, nupeaks,
                                         voxel_step_size,
                                         args.theta)
        if len(streamline) > 1:
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
