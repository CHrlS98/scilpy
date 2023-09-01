#!/usr/bin/env python3
"""
Script for scoring a predicted peak on an expected peaks image.
"""
import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd

from scilpy.segment.voxlabel import classify_peaks_asym

STRAIGHT_EPSILON = -0.9999
CLASSES = {
    'no_peak': 0,
    'single_peak': 1,
    'straight': 2,
    'bending': 2.5,
    'branching': 3,
    'crossing_sym': 4,
    'crossing_asym': 4.5,
    'others': 5
}


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_expected')
    p.add_argument('out_csv')
    p.add_argument('--in_mask')
    p.add_argument('--in_predicted', required=True, nargs='+')

    p.add_argument('--in_expected_labels')
    p.add_argument('--out_expected_labels')
    p.add_argument('--tag')
    p.add_argument('--bend_tol', default=2.0, type=float)

    p.add_argument('--write_mode', choices=['append', 'write'],
                   help="File write mode. `append` will append score to\n"
                        "existing file (won't write csv header). `write` will\n"
                        "create a new file (and write the file header).")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # once
    expected_im = nib.load(args.in_expected)
    expected_peaks = np.reshape(expected_im.get_fdata(),
                                np.append(expected_im.shape[:3], (-1, 3)))
    if args.in_mask:
        mask = nib.load(args.in_mask).get_fdata().astype(bool)
    else:
        mask = None

    expected_labels = classify_peaks_asym(expected_peaks, args.bend_tol)
    if mask is not None:
        unique, count = np.unique(expected_labels[mask], return_counts=True)
    else:
        unique, count = np.unique(expected_labels, return_counts=True)
    print('Labels: ', unique)
    print('Counts: ', count)

    results = {
        'tag' : [],
        'no_peak': [],
        'single_peak': [],
        'straight': [],
        'bending': [],
        'branching': [],
        'crossing_sym': [],
        'crossing_asym': [],
        'others': []
    }

    # for all images
    for it, in_predicted in enumerate(args.in_predicted):
        print('Processing subject {} out of {}'.format(it+1, len(args.in_predicted)))
        predicted_im = nib.load(in_predicted)
        predicted_peaks = np.reshape(predicted_im.get_fdata(),
                                    np.append(predicted_im.shape[:3], (-1, 3)))

        # 1st. Classify voxels.
        predicted_labels = classify_peaks_asym(predicted_peaks, args.bend_tol)

        # 2nd. For each voxel class, count TP rate.
        tag = os.path.basename(in_predicted)
        results['tag'].append(tag)

        for label, ind in CLASSES.items():
            prediction_match = np.logical_and(expected_labels == ind, predicted_labels == ind)
            if mask is not None:
                prediction_match[~mask] = 0
            nb_hits = np.count_nonzero(prediction_match)
            if nb_hits > 0:
                if mask is not None:
                    rate = float(nb_hits) / float(np.count_nonzero(expected_labels[mask] == ind))
                else:  # no mask
                    rate = float(nb_hits) / float(np.count_nonzero(expected_labels == ind))
            else:
                rate = 0.0
            results[label].append(rate)

    table = pd.DataFrame(results)
    table.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
