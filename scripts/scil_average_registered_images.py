#!/usr/bin/env python3
# -*- coding:uft-8 -*-
"""
Given a bunch of registered nifti images with corresponding x, y, z dimensions,
compute an average map. The number of dimensions for the last dimension is
given by the maximum number of values across all images.
"""
import argparse
import nibabel as nib
import numpy as np

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volumes', nargs='+', help='Input images to average.')
    p.add_argument('--out', help='Output image.', required=True)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_volumes)
    assert_outputs_exist(parser, args, args.out)

    # first image
    img = nib.load(args.in_volumes[0])
    sum_of_vols = img.get_fdata(dtype=np.float32)[..., :10]

    for fname in args.in_volumes[1:]:
        img = nib.load(fname)
        img_data = img.get_fdata(dtype=np.float32)[..., :10]
        if img_data.shape[:3] != sum_of_vols.shape[:3]:
            raise ValueError('Dimension mismatch for image {}'.format(fname))
        if img_data.shape[-1] > sum_of_vols.shape[-1]:
            img_data[..., :sum_of_vols.shape[-1]] += sum_of_vols
            sum_of_vols = img_data
        else:  # sum_of_vols.shape[-1] >= img.shape[-1]
            sum_of_vols[..., :img_data.shape[-1]] += img_data

    sum_of_vols /= len(args.in_volumes)
    nib.save(nib.Nifti1Image(sum_of_vols, img.affine), args.out)


if __name__ == '__main__':
    main()
