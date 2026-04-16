#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Draw a spherical region of interest on a reference image centered on `voxel_position`
with radius `radius`. This script can be useful for manually defining seeding regions
for tractography.
"""
import argparse
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, add_overwrite_arg

import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input reference image onto which the ROI is drawn.')
    p.add_argument('out_image',
                   help='Output binary image with the ROI.')
    p.add_argument('voxel_position', type=int, nargs=3,
                   help='Center of the ROI in voxel coordinates.')
    p.add_argument('radius', type=int,
                   help='Radius of the ROI in voxels.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_image)

    ref_image = nib.load(args.in_image)
    output = np.moveaxis(np.indices(ref_image.shape, sparse=False), 0, -1)

    center_pos = np.array(args.voxel_position).reshape((1, 1, 1, 3))
    roi = np.linalg.norm(output - center_pos, axis=-1) <= args.radius

    nib.save(nib.Nifti1Image(roi.astype(np.uint8), ref_image.affine),
             args.out_image)


if __name__ == '__main__':
    main()
