#!/usr/bin/env python3
# -*- coding:uft-8 -*-
"""
Given a bunch of registered nifti images with corresponding dimensions,
compute an average map.
"""
import argparse
import nibabel as nib
import numpy as np

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volumes', nargs='+', help='Input images to average.')
    p.add_argument('--out', help='Output image.', required=True)

    p.add_argument('--summarize_metric', choices=['mean', 'vote'],
                   default='mean', help='Metric used to summarize values.')

    add_overwrite_arg(p)
    return p


def compute_mean_image(args):
    # first image
    img = nib.load(args.in_volumes[0])
    sum_of_vols = img.get_fdata()

    for fname in args.in_volumes[1:]:
        img = nib.load(fname)
        img_data = img.get_fdata()
        if img_data.shape[:3] != sum_of_vols.shape[:3]:
            raise ValueError('Dimension mismatch for image {}'.format(fname))
        sum_of_vols += img_data

    sum_of_vols /= len(args.in_volumes)
    return sum_of_vols


def classify_majority_vote(args):
    img = nib.load(args.in_volumes[0])
    data = img.get_fdata().astype(int)
    vshape = data.shape[:3]

    # flat histogram
    bins = np.zeros(np.append(data.shape, 11), dtype=int)
    bins = np.reshape(bins, (-1, 11))

    # process 1st image
    bins[np.arange(len(bins)), data.flatten()] += 1

    for fname in args.in_volumes[1:]:
        img = nib.load(fname)
        data = img.get_fdata().astype(int)
        if data.shape[:3] != vshape:
            raise ValueError('Dimension mismatch for image {}'.format(fname))
        bins[np.arange(len(bins)), data.flatten()] += 1

    # reshape image
    bins = bins.reshape(np.append(data.shape, 11))

    voted = np.argmax(bins, axis=-1)
    return voted


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_volumes)
    assert_outputs_exist(parser, args, args.out)

    ref_im = nib.load(args.in_volumes[0])

    # first image
    if args.summarize_metric == 'mean':
        output = compute_mean_image(args)
    else:
        output = classify_majority_vote(args)

    nib.save(nib.Nifti1Image(output.astype(ref_im.get_data_dtype()), ref_im.affine), args.out)


if __name__ == '__main__':
    main()
