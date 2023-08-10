#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from dipy.io import read_bvals_bvecs
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Input DW image.')
    p.add_argument('in_bval',
                   help='Input b-value file.')
    p.add_argument('out_dwi',
                   help='Output corrected DW image.')

    p.add_argument('--show', action='store_true',
                   help='Show plot of mean intensities before/after '
                        'correction.')
    p.add_argument('--mask',
                   help='Only voxels inside the mask will be used for\n'
                        'bias estimation.')
    p.add_argument('--tolerance', '-t', type=int, default=20,
                   help='The tolerated gap between the b-values to'
                        ' extract\nand the actual b-values.')
    p.add_argument('--degree', type=int, default=2,
                   help='Degree of the polynomial curve fitted to the data.')
    add_overwrite_arg(p)
    return p


def phi(x, degree):
    return np.vstack([x**i for i in range(degree + 1)]).T


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval], args.mask)
    assert_outputs_exist(parser, args, args.out_dwi)

    bvals, _ = read_bvals_bvecs(args.in_bval, None)

    # extract b-values
    b0_mask = bvals < args.tolerance
    b0_indices = np.flatnonzero(b0_mask)
    in_dwi = nib.load(args.in_dwi)
    dwi = in_dwi.get_fdata(dtype=float)
    # print(dwi.shape)

    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        mask = np.ones(dwi.shape[:-1], dtype=bool)

    # b0s = np.take_along_axis(dwi, b0_indices, axis=-1)
    # print(b0_indices)
    mean_dwi = np.mean(dwi[mask], axis=0)

    b0s =  dwi[..., b0_indices]

    mean_b0s = np.mean(b0s[mask], axis=0)
    mean_b0 = np.mean(mean_b0s)

    # regression
    # f(x) = [a, b]^T * [x, 1]
    phi_x = phi(b0_indices, args.degree)
    psi = mean_b0s.reshape((-1, 1))
    w = np.linalg.inv(phi_x.T.dot(phi_x)).dot(phi_x.T).dot(psi)  # (2, 1)

    phi_all = phi(np.arange(len(b0_mask)), args.degree)
    pred_curve = w.T.dot(phi_all.T)

    rectified_dwi = dwi / pred_curve.reshape((1, 1, 1, -1)) * mean_b0
    mean_rect_dwi = np.mean(rectified_dwi[mask], axis=0)

    nib.save(nib.Nifti1Image(rectified_dwi, in_dwi.affine), args.out_dwi)
    
    if args.show:
        # plot b0 images
        plt.plot(b0_indices, mean_b0s, '.-', alpha=0.5, label='Input b0')
        plt.plot(b0_indices, mean_rect_dwi[b0_indices], '.-', alpha=0.5,
                 label='Rectified b0')

        all_indices = np.arange(len(b0_mask))
        plt.plot(all_indices, pred_curve.squeeze(), '--', alpha=0.5,
                label='Signal drift curve (degree {})'.format(args.degree))

        dw_indices = all_indices[~b0_mask]
        plt.plot(dw_indices, mean_dwi[dw_indices], '.', label='Input DWI')
        plt.plot(dw_indices, mean_rect_dwi[dw_indices], '.',
                 label='Recitified DWI')

        plt.title('Mean intensities of DWI images')
        plt.ylabel('Mean intensity')
        plt.xlabel('Index')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
