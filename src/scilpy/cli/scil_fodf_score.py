#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare two (f)ODF images by computing the mean absolute difference in SF amplitudes
at each voxel. The output is a scalar map of the same shape as the input images.

If peaks are provided, the script will also compute the angular error between both peaks images.
"""

import argparse
import logging

from dipy.data import SPHERE_FILES, get_sphere
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import nibabel as nib
import numpy as np
from tqdm import tqdm

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_headers_compatible, assert_inputs_exist,
                             assert_outputs_exist, parse_sh_basis_arg,
                             add_json_args)
from scilpy.reconst.sh import convert_sh_to_sf
from scilpy.version import version_string
import json


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_sh',
                   help='Path of the input ODF image.')
    p.add_argument('in_peaks_ref',
                   help='Path to a reference peaks image. When provided, the script will also\n'
                        'report the average density of in_sh along the reference peaks. The higher\n'
                        'this is, the better we are at capturing the expected peaks directions.')
    p.add_argument('out_json',
                   help='Output JSON file to save the computed metrics.')

    p.add_argument('--peaks_estimate',
                   help='Path to a peaks image estimated from the input ODFs. When supplied,\n'
                        'the script will compute the angular error between the estimated\n'
                        'and reference peaks.')
    p.add_argument('--out_angular_error',
                   help='Output filename for the angular error map between estimated and reference peaks.\n'
                        'Only used if --peaks_estimate is provided.')
    p.add_argument('--out_sf_along_peaks',
                   help='Output filename for the average SF amplitude along reference peaks.\n'
                        'Only used if --ref_peaks is provided.')
    p.add_argument('--rel_threshold', type=float, default=0.1,
                   help='Relative threshold for considering that we captured a peak direction.\n'
                        'Only used if --ref_peaks is provided. [%(default)s]')
    
    p.add_argument('--sphere', default='repulsion724',
                   choices=SPHERE_FILES.keys(),
                   help='Sphere to use for ODF evaluation. [%(default)s].')
    add_verbose_arg(p)
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    add_processes_arg(p)
    add_json_args(p)
    return p


def _compute_angular_error(est_peaks, ref_peaks):
    # normalize estimated peaks to unit length
    norm = np.linalg.norm(est_peaks, axis=-1, keepdims=True)
    est_peaks = np.divide(est_peaks, norm, where=norm > 0)
    angular_error = np.zeros(est_peaks.shape[:3])
    for peak_i in range(ref_peaks.shape[-2]):
        ref_peaks_i = ref_peaks[..., peak_i, :]
        valid_i = np.linalg.norm(ref_peaks_i, axis=-1) > 0
        if not np.any(valid_i):
            continue  # no valid peaks (should not happen)

        # compare this peak to all estimated peaks
        dot_prod = np.abs(np.sum(est_peaks * ref_peaks_i[..., None, :], axis=-1))
        dot_prod = np.clip(dot_prod, -1, 1)  # clip for numerical stability

        _angular_error = np.rad2deg(np.arccos(dot_prod))
        _angular_error = np.min(_angular_error, axis=-1)  # take the minimum error across all estimated peaks
        _angular_error[~valid_i] = 0  # ignore voxels without valid peaks in the reference
        angular_error = np.maximum(angular_error, _angular_error)  # take the maximum error across all peaks
    return angular_error


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_sh, args.in_peaks_ref])
    assert_outputs_exist(parser, args, args.out_json, [args.out_sf_along_peaks, args.out_angular_error])

    assert_headers_compatible(parser, [args.in_sh, args.in_peaks_ref])

    sh_im = nib.load(args.in_sh)
    sh = sh_im.get_fdata(dtype=np.float32)

    basis, legacy = parse_sh_basis_arg(args)
    sh_order = order_from_ncoef(sh.shape[-1])

    sphere = get_sphere(name=args.sphere)
    B = sh_to_sf_matrix(sphere, sh_order_max=sh_order,
                        basis_type=basis, legacy=legacy,
                        return_inv=False)

    # save all output metrics to this dictionary which we'll save as json at the end of the script
    metrics_dict = {}

    ref_peaks_im = nib.load(args.in_peaks_ref)
    ref_peaks = ref_peaks_im.get_fdata(dtype=np.float32)
    ref_peaks = ref_peaks.reshape(ref_peaks.shape[:-1] + (-1, 3))
    norm = np.linalg.norm(ref_peaks, axis=-1)
    ref_peaks = np.divide(ref_peaks, norm[..., None], where=norm[..., None] > 0)  # normalize
    valid = norm > 0  # valid is 4D
    mask = np.sum(valid, axis=-1) > 0  # 3D mask of voxels with at least one valid peak
    nufo = np.count_nonzero(valid, axis=-1)
    nufo[~mask] = 0

    if args.peaks_estimate is not None:
        est_peaks_im = nib.load(args.peaks_estimate)
        est_peaks = est_peaks_im.get_fdata(dtype=np.float32)
        est_peaks = est_peaks.reshape(est_peaks.shape[:3] + (-1, 3))
        max_angular_error = _compute_angular_error(est_peaks, ref_peaks)
        if args.out_angular_error is not None:
            nib.save(nib.Nifti1Image(max_angular_error.astype(np.float32), sh_im.affine),
                     args.out_angular_error)
        metrics_dict['mean_max_angular_error'] = np.mean(max_angular_error[mask])
        for i in range(1, nufo.max() + 1):
            count = np.count_nonzero(nufo == i)
            if count > 0:
                metrics_dict[f'mean_max_angular_error_nufo_{i}'] = np.mean(max_angular_error[nufo == i])

    peaks1d = ref_peaks[valid]
    peaks1d_to_sph_ind = np.zeros((peaks1d.shape[0],), dtype=int)
    max_dot = np.zeros((peaks1d.shape[0],))
    for i, direction in enumerate(tqdm(sphere.vertices, desc='Matching peaks to sphere vertices')):
        dot_prod_1d = np.abs(peaks1d.dot(direction))
        update = dot_prod_1d > max_dot
        max_dot[update] = dot_prod_1d[update]
        peaks1d_to_sph_ind[update] = i
    peaks_to_sph_ind = np.full(ref_peaks.shape[:-1], -1, dtype=int)
    peaks_to_sph_ind[valid] = peaks1d_to_sph_ind

    min_amplitude_along_peaks = []
    for slice_i, sh_i in enumerate(tqdm(sh, desc='Processing slices')):
        sf = sh_i.dot(B)
        sf_max = np.max(sf, axis=-1)

        # L-max normalization of SF amplitudes for fair comparison
        sf = np.divide(sf, sf_max[..., None], where=sf_max[..., None] > 0)

        peaks_to_sph_ind_i = peaks_to_sph_ind[slice_i]  # (ny, nz, npeaks) as sf and sf_ref
        min_amplitudes_slice_i = np.zeros(peaks_to_sph_ind_i.shape[:-1])
        for_overwrite = peaks_to_sph_ind_i[..., 0] != -1
        # initialize to a value > 1 so that any valid amplitude along peaks will be kept as minimum
        # works because we know that the valid mask will always be a subset of the valid mask of previous peak
        min_amplitudes_slice_i[for_overwrite] = 2

        for peaks_i in range(peaks_to_sph_ind_i.shape[-1]):
            amplitudes =  np.take_along_axis(sf, peaks_to_sph_ind_i[..., peaks_i][..., None], axis=-1).squeeze()
            amplitudes[amplitudes < 0] = 0  # we only care about positive amplitudes along peaks directions
            valid = peaks_to_sph_ind_i[..., peaks_i] != -1
            if not np.any(valid):
                continue # no valid peaks in this slice
            min_amplitudes_slice_i[valid] = np.minimum(min_amplitudes_slice_i[valid], amplitudes[valid])
        min_amplitude_along_peaks.append(min_amplitudes_slice_i)

    min_amplitude_along_peaks = np.array(min_amplitude_along_peaks)
    min_amplitude_along_peaks[~mask] = 0

    metrics_dict['mean_min_amplitude_along_peaks'] = np.mean(min_amplitude_along_peaks[mask])
    metrics_dict['pct_peaks_captured'] = \
        np.count_nonzero(min_amplitude_along_peaks[mask] > args.rel_threshold) / float(np.count_nonzero(mask))

    for i in range(1, nufo.max() + 1):
        count = np.count_nonzero(nufo == i)
        if count > 0:
            metrics_dict[f'mean_min_amplitude_along_peaks_nufo_{i}'] = np.mean(min_amplitude_along_peaks[nufo == i])
            metrics_dict[f'pct_peaks_captured_nufo_{i}'] = \
                np.count_nonzero(min_amplitude_along_peaks[nufo == i] > args.rel_threshold) / float(count)

    if args.out_sf_along_peaks is not None:
        nib.save(nib.Nifti1Image(min_amplitude_along_peaks, sh_im.affine), args.out_sf_along_peaks)

    with open(args.out_json, 'w') as outfile:
        json.dump(metrics_dict, outfile,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()

