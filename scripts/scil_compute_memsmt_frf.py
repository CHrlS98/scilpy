#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to estimate response functions for multi-encoding multi-shell
multi-tissue (memsmt) constrained spherical deconvolution. In order to operate,
the script only needs the data from one type of b-tensor encoding. However,
giving only a spherical one will not produce good fiber response functions, as
it only probes spherical shapes. As for planar encoding, it should technically
work alone, but seems to be very sensitive to noise and is yet to be properly
documented. We thus suggest to always use at least the linear encoding, which
will be equivalent to standard multi-shell multi-tissue if used alone, in
combinaison with other encodings. Not that custom encodings are not yet
supported, except for the cigar shape (b_delta = 0.5).

The script computes a response function for white-matter (wm),
gray-matter (gm), csf and the mean b=0.

In the wm, we compute the response function in each voxels where
the FA is superior at threshold_fa_wm.

In the gm (or csf), we compute the response function in each voxels where
the FA is below at threshold_fa_gm (or threshold_fa_csf) and where
the MD is below threshold_md_gm (or threshold_md_csf).

Based on P. Karan et al., Enabling constrained spherical deconvolution and
diffusional variance decomposition with tensor-valued diffusion MRI.
BioRxiv (2020) https://www.biorxiv.org/content/10.1101/2021.04.07.438845v1
"""
# Add Karan et al., 2021 when published.

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.utils import extract_affine
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_force_b0_arg,
                             add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.frf import compute_msmt_frf
from scilpy.utils.bvec_bval_tools import extract_dwi_shell
from scilpy.reconst.b_tensor_utils import generate_btensor_input


def buildArgsParser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('out_wm_frf',
                   help='Path to the output WM frf file, in .txt format.')
    p.add_argument('out_gm_frf',
                   help='Path to the output GM frf file, in .txt format.')
    p.add_argument('out_csf_frf',
                   help='Path to the output CSF frf file, in .txt format.')

    p.add_argument('--in_dwi_linear', metavar='file', default=None,
                   help='Path of the linear input diffusion volume.')
    p.add_argument('--in_bval_linear', metavar='file', default=None,
                   help='Path of the linear bvals file, in FSL format.')
    p.add_argument('--in_bvec_linear', metavar='file', default=None,
                   help='Path of the linear bvecs file, in FSL format.')
    p.add_argument('--in_dwi_planar', metavar='file', default=None,
                   help='Path of the planar input diffusion volume.')
    p.add_argument('--in_bval_planar', metavar='file', default=None,
                   help='Path of the planar bvals file, in FSL format.')
    p.add_argument('--in_bvec_planar', metavar='file', default=None,
                   help='Path of the planar bvecs file, in FSL format.')
    p.add_argument('--in_dwi_spherical', metavar='file', default=None,
                   help='Path of the spherical input diffusion volume.')
    p.add_argument('--in_bval_spherical', metavar='file', default=None,
                   help='Path of the spherical bvals file, in FSL format.')
    p.add_argument('--in_bvec_spherical', metavar='file', default=None,
                   help='Path of the spherical bvecs file, in FSL format.')
    p.add_argument('--in_dwi_custom', metavar='file', default=None,
                   help='Path of the custom input diffusion volume.')
    p.add_argument('--in_bval_custom', metavar='file', default=None,
                   help='Path of the custom bvals file, in FSL format.')
    p.add_argument('--in_bvec_custom', metavar='file', default=None,
                   help='Path of the custom bvecs file, in FSL format.')
    p.add_argument('--in_bdelta_custom', type=float, choices=[0, 1, -0.5, 0.5],
                   help='Value of the b_delta for the custom encoding.')

    p.add_argument('--mask',
                   help='Path to a binary mask. Only the data inside the mask '
                        'will be used for\ncomputations and reconstruction. '
                        'Useful if no tissue masks are available.')
    p.add_argument('--mask_wm',
                   help='Path to the input WM mask file, used to improve the'
                        ' final WM frf mask.')
    p.add_argument('--mask_gm',
                   help='Path to the input GM mask file, used to improve the '
                        'final GM frf mask.')
    p.add_argument('--mask_csf',
                   help='Path to the input CSF mask file, used to improve the'
                        ' final CSF frf mask.')

    p.add_argument('--fa_thr_wm',
                   default=0.7, type=float,
                   help='If supplied, use this threshold to select single WM '
                        'fiber voxels from the FA inside the WM mask defined '
                        ' by mask_wm. Each voxel above this threshold will '
                        'be selected. [%(default)s]')
    p.add_argument('--fa_thr_gm',
                   default=0.2, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the FA inside the GM mask defined by mask_gm. '
                        'Each voxel below this threshold will be selected.'
                        ' [%(default)s]')
    p.add_argument('--fa_thr_csf',
                   default=0.1, type=float,
                   help='If supplied, use this threshold to select CSF voxels '
                        'from the FA inside the CSF mask defined by mask_csf. '
                        'Each voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--md_thr_gm',
                   default=0.0007, type=float,
                   help='If supplied, use this threshold to select GM voxels '
                        'from the MD inside the GM mask defined by mask_gm. '
                        'Each voxel below this threshold will be selected. '
                        '[%(default)s]')
    p.add_argument('--md_thr_csf',
                   default=0.003, type=float,
                   help='If supplied, use this threshold to select CSF '
                        'voxels from the MD inside the CSF mask defined by '
                        'mask_csf. Each voxel below this threshold will be'
                        ' selected. [%(default)s]')

    p.add_argument('--min_nvox',
                   default=100, type=int,
                   help='Minimal number of voxels needed for each tissue masks'
                        ' in order to proceed to frf estimation. '
                        '[%(default)s]')
    p.add_argument('--tolerance',
                   type=int, default=20,
                   help='The tolerated gap between the b-values to '
                        'extract and the current b-value. [%(default)s]')
    p.add_argument('--dti_bval_limit',
                   type=int, default=1200,
                   help='The highest b-value taken for the DTI model. '
                        '[%(default)s]')
    p.add_argument('--roi_radii',
                   default=[20], nargs='+', type=int,
                   help='If supplied, use those radii to select a cuboid roi '
                        'to estimate the response functions. The roi will be '
                        'a cuboid spanning from the middle of the volume in '
                        'each direction with the different radii. The type is '
                        'either an int (e.g. --roi_radii 10) or an array-like '
                        '(3,) (e.g. --roi_radii 20 30 10). [%(default)s]')
    p.add_argument('--roi_center',
                   metavar='tuple(3)', nargs=3, type=int,
                   help='If supplied, use this center to span the cuboid roi '
                        'using roi_radii. [center of the 3D volume] '
                        '(e.g. --roi_center 66 79 79)')

    p.add_argument('--wm_frf_mask',
                   metavar='file', default='',
                   help='Path to the output WM frf mask file, the voxels used '
                        'to compute the WM frf.')
    p.add_argument('--gm_frf_mask',
                   metavar='file', default='',
                   help='Path to the output GM frf mask file, the voxels used '
                        'to compute the GM frf.')
    p.add_argument('--csf_frf_mask',
                   metavar='file', default='',
                   help='Path to the output CSF frf mask file, the voxels '
                        'used to compute the CSF frf.')

    p.add_argument('--frf_table',
                   metavar='file', default='',
                   help='Path to the output frf table file. Saves the frf for '
                        'each b-value, in .txt format.')

    add_force_b0_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [],
                        optional=[args.in_dwi_linear, args.in_bval_linear,
                                  args.in_bvec_linear,
                                  args.in_dwi_planar, args.in_bval_planar,
                                  args.in_bvec_planar,
                                  args.in_dwi_spherical,
                                  args.in_bval_spherical,
                                  args.in_bvec_spherical])
    assert_outputs_exist(parser, args, [args.out_wm_frf, args.out_gm_frf,
                                        args.out_csf_frf])

    input_files = [args.in_dwi_linear, args.in_dwi_planar,
                   args.in_dwi_spherical, args.in_dwi_custom]
    bvals_files = [args.in_bval_linear, args.in_bval_planar,
                   args.in_bval_spherical, args.in_bval_custom]
    bvecs_files = [args.in_bvec_linear, args.in_bvec_planar,
                   args.in_bvec_spherical, args.in_bvec_custom]
    b_deltas_list = [1.0, -0.5, 0, args.in_bdelta_custom]

    for i in range(4):
        enc = ["linear", "planar", "spherical", "custom"]
        if (input_files[i] is None and bvals_files[i] is None
           and bvecs_files[i] is None):
            inclusive = 1
            if i == 3 and args.in_bdelta_custom is not None:
                inclusive = 0
        elif (input_files[i] is not None and bvals_files[i] is not None
              and bvecs_files[i] is not None):
            inclusive = 1
            if i == 3 and args.in_bdelta_custom is None:
                inclusive = 0
        else:
            inclusive = 0
        if inclusive == 0:
            msg = """All of in_dwi, bval and bvec files are mutually needed
                  for {} encoding."""
            raise ValueError(msg.format(enc[i]))

    affine = extract_affine(input_files)

    if len(args.roi_radii) == 1:
        roi_radii = args.roi_radii[0]
    elif len(args.roi_radii) == 2:
        parser.error('--roi_radii cannot be of size (2,).')
    else:
        roi_radii = args.roi_radii
    roi_center = args.roi_center

    tol = args.tolerance
    dti_lim = args.dti_bval_limit
    force_b0_thr = args.force_b0_threshold

    gtab, data, ubvals, ubdeltas = generate_btensor_input(input_files,
                                                          bvals_files,
                                                          bvecs_files,
                                                          b_deltas_list,
                                                          force_b0_thr,
                                                          tol=tol)

    if not np.all(ubvals <= dti_lim):
        if np.sum(ubdeltas == 1) > 0:
            dti_ubvals = ubvals[ubdeltas == 1]
        elif np.sum(ubdeltas == -0.5) > 0:
            dti_ubvals = ubvals[ubdeltas == -0.5]
        elif np.sum(ubdeltas == args.in_bdelta_custom) > 0:
            dti_ubvals = ubvals[ubdeltas == args.in_bdelta_custom]
        else:
            raise ValueError("No encoding available for DTI.")
        vol = nib.Nifti1Image(data, affine)
        outputs = extract_dwi_shell(vol, gtab.bvals, gtab.bvecs,
                                    dti_ubvals[dti_ubvals <= dti_lim],
                                    tol=1)
        indices_dti, data_dti, bvals_dti, bvecs_dti = outputs
        # gtab_dti = gradient_table(np.squeeze(bvals_dti), bvecs_dti,
        #                           btens=gtab.btens[indices_dti])
        bvals_dti = np.squeeze(bvals_dti)
        btens_dti = gtab.btens[indices_dti]
    else:
        data_dti = None
        bvals_dti = None
        bvecs_dti = None
        btens_dti = None

    mask = None
    if args.mask is not None:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
    mask_wm = None
    mask_gm = None
    mask_csf = None
    if args.mask_wm:
        mask_wm = get_data_as_mask(nib.load(args.mask_wm), dtype=bool)
    if args.mask_gm:
        mask_gm = get_data_as_mask(nib.load(args.mask_gm), dtype=bool)
    if args.mask_csf:
        mask_csf = get_data_as_mask(nib.load(args.mask_csf), dtype=bool)

    responses, frf_masks = compute_msmt_frf(data, gtab.bvals, gtab.bvecs,
                                            btens=gtab.btens,
                                            data_dti=data_dti,
                                            bvals_dti=bvals_dti,
                                            bvecs_dti=bvecs_dti,
                                            btens_dti=btens_dti,
                                            mask=mask, mask_wm=mask_wm,
                                            mask_gm=mask_gm, mask_csf=mask_csf,
                                            fa_thr_wm=args.fa_thr_wm,
                                            fa_thr_gm=args.fa_thr_gm,
                                            fa_thr_csf=args.fa_thr_csf,
                                            md_thr_gm=args.md_thr_gm,
                                            md_thr_csf=args.md_thr_csf,
                                            min_nvox=args.min_nvox,
                                            roi_radii=roi_radii,
                                            roi_center=roi_center,
                                            tol=0,
                                            force_b0_threshold=force_b0_thr)

    masks_files = [args.wm_frf_mask, args.gm_frf_mask, args.csf_frf_mask]
    for mask, mask_file in zip(frf_masks, masks_files):
        if mask_file:
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), vol.affine),
                     mask_file)

    frf_out = [args.out_wm_frf, args.out_gm_frf, args.out_csf_frf]

    for frf, response in zip(frf_out, responses):
        np.savetxt(frf, response)

    if args.frf_table:
        if ubvals[0] < tol:
            bvals = ubvals[1:]
        else:
            bvals = ubvals
        response_csf = responses[2]
        response_gm = responses[1]
        response_wm = responses[0]
        iso_responses = np.concatenate((response_csf[:, :3],
                                        response_gm[:, :3]), axis=1)
        responses = np.concatenate((iso_responses, response_wm[:, :3]), axis=1)
        frf_table = np.vstack((bvals, responses.T)).T
        np.savetxt(args.frf_table, frf_table)


if __name__ == "__main__":
    main()
