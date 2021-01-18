# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert volume fractions map as given by scil_compute_msmt_fodf.py
to probabilities using the softmax function.
"""

import argparse
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.utils import load_matrix_in_any_format
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from scipy.special import softmax


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_wm_fodf', help='Input WM fODF SH file.')
    p.add_argument('in_vf', help='Input volume fractions file.')
    p.add_argument('in_fodf_th',
                   help='Path to max fODF amplitude in ventricles text file.')
    p.add_argument('out_vf', help='Output normalized volume fractions map.')

    p.add_argument('--save_rgb', action='store_true',
                   help='Save additionnal RGB image.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_vf, args.in_wm_fodf, args.in_fodf_th])
    assert_outputs_exist(parser, args, args.out_vf)

    vf_img = nib.load(args.in_vf)
    vf = vf_img.get_fdata()

    vf_mask = np.sum(vf, axis=-1, keepdims=True) > 0.
    vf_norm = softmax(vf, axis=-1) * vf_mask
    nib.save(nib.Nifti1Image(vf_norm, vf_img.affine), args.out_vf)

    if args.save_rgb:
        rgb = (vf_norm * 255).astype(np.uint8)
        ext_pos = args.out_vf.find('.nii')
        rgb_fname = args.out_vf[:ext_pos] + '_rgb' + args.out_vf[ext_pos:]
        nib.save(nib.Nifti1Image(rgb, vf_img.affine), rgb_fname)


if __name__ == '__main__':
    main()
