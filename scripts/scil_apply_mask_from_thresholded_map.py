#!/usr/bin/env python3

"""
Apply a mask to a given image from a thresholded map.
"""

import argparse
import nibabel as nib
import numpy as np
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_img', help='Input image to mask.')
    p.add_argument('out_img', help='Output masked image.')
    p.add_argument('in_map', help='Input map used for masking.')
    p.add_argument('threshold', type=float, help='Threshold used for masking.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Safety checks for inputs
    assert_inputs_exist(parser, [args.in_img, args.in_map])

    # Safety checks for outputs
    assert_outputs_exist(parser, args, args.out_img)

    # Load inputs
    in_img = nib.load(args.in_img)
    in_map_img = nib.load(args.in_map)
    mask = in_map_img.get_fdata() > args.threshold
    out_img = mask * in_img.get_fdata()

    nib.save(nib.Nifti1Image(out_img, in_img.affine), args.out_img)


if __name__ == '__main__':
    main()
