#!/usr/bin/env python3
"""
MAP-MRI
"""
import argparse
import nibabel as nib

from dipy.reconst import mapmri
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from scilpy.reconst.mapmri import fit_from_model
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.io.image import get_data_as_mask


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi')
    p.add_argument('in_bval')
    p.add_argument('in_bvec')
    p.add_argument('out_mapmri')

    p.add_argument('--mask', help='Optional input mask.')

    p.add_argument('--big_delta', type=float,
                   help='Acquisition big delta parameter (seconds).')
    p.add_argument('--small_delta', type=float,
                   help='Acquisition small delta parameter (seconds).')

    add_processes_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec], args.mask)
    assert_outputs_exist(parser, args, args.out_mapmri)

    dwi_im = nib.load(args.in_dwi)
    dwi = dwi_im.get_fdata()

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    mask = None
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    # From the DIPY tutorials:
    # For the values of the q-space indices to make sense it is necessary to
    # explicitly state the big_delta and small_delta parameters in the gradient table.
    gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                          big_delta=args.big_delta,
                          small_delta=args.small_delta)

    radial_order = 6
    map_model = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                   laplacian_regularization=True,
                                   laplacian_weighting=.05,
                                   positivity_constraint=True)

    mapmri_fit = fit_from_model(map_model, dwi, mask=mask,
                                nbr_processes=args.nbr_processes)
    mapmri_coeffs = mapmri_fit.mapmri_coeff
    nib.save(nib.Nifti1Image(mapmri_coeffs, dwi_im.affine), args.out_mapmri)


if __name__ == '__main__':
    main()
