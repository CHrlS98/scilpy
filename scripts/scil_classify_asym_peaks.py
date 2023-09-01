#!/usr/bin/env python3
"""
Script for labelling asymmetric peak configurations.
"""
import argparse
import os
import nibabel as nib
import numpy as np

from scilpy.segment.voxlabel import classify_peaks_asym


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_peaks')
    p.add_argument('out_labels')

    p.add_argument('--bend_tol', default=2.0, type=float)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # once
    peaks_im = nib.load(args.in_peaks)
    peaks = np.reshape(peaks_im.get_fdata(),
                       np.append(peaks_im.shape[:3],
                                 (-1, 3)))
    labels = classify_peaks_asym(peaks, args.bend_tol)

    unique, count = np.unique(labels, return_counts=True)
    print(count)

    nib.save(nib.Nifti1Image(labels.astype(np.float32), peaks_im.affine),
             args.out_labels)


if __name__ == '__main__':
    main()
