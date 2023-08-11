#!/usr/bin/env python3
"""
Script for scoring a predicted peak on an expected peaks image.
"""
import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd

STRAIGHT_EPSILON = -0.9999
CLASSES = {
    'no_peak': 0,
    'single_peak': 1,
    'straight': 2,
    'bending': 3,
    'branching': 4,
    'crossing': 5,
    'others': 6
}


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_expected')
    p.add_argument('out_csv')
    p.add_argument('--in_predicted', required=True, nargs='+')

    p.add_argument('--in_expected_labels')
    p.add_argument('--out_expected_labels')
    p.add_argument('--tag')

    p.add_argument('--write_mode', choices=['append', 'write'],
                   help="File write mode. `append` will append score to\n"
                        "existing file (won't write csv header). `write` will\n"
                        "create a new file (and write the file header).")
    return p


def classify_peaks(peaks):
    # classes are:
    # 0. No peaks
    # 1. 1-peak
    # 2. Straight fiber
    # 3. Bending fiber
    # 4. Branching
    # 5. Crossing
    # 6. Others
    # (it becomes hard to make sense of
    #  configurations higher than crossing)
    peak_norms = np.linalg.norm(peaks, axis=-1)
    nufid = np.count_nonzero(peak_norms, axis=-1)
    labels = np.zeros_like(nufid)

    # normalize peaks for later
    peaks[peak_norms > 0] /= peak_norms[peak_norms > 0][..., None]

    # label "obvious" configurations
    labels[nufid == 1] = CLASSES['single_peak']
    labels[nufid == 3] = CLASSES['branching']
    labels[nufid == 4] = CLASSES['crossing']

    # identify two-direction voxels (straight and bending)
    idx, idy, idz = np.nonzero(nufid == 2)
    for ind in zip(idx, idy, idz):
        p0 = peaks[ind][0]
        p1 = peaks[ind][1]
        dot = p0.dot(p1)
        if dot < STRAIGHT_EPSILON:
            labels[ind] = CLASSES['straight']
        else:
            labels[ind] = CLASSES['bending']
    labels[nufid > 4] = CLASSES['others']
    return labels


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # once
    expected_im = nib.load(args.in_expected)
    expected_peaks = np.reshape(expected_im.get_fdata(),
                                np.append(expected_im.shape[:3], (-1, 3)))
    expected_labels = classify_peaks(expected_peaks)
    if args.out_expected_labels:
        nib.save(nib.Nifti1Image(expected_labels.astype(np.uint8),
                                 expected_im.affine),
                 args.out_expected_labels)

    results = {
        'tag' : [],
        'no_peak': [],
        'single_peak': [],
        'straight': [],
        'bending': [],
        'branching': [],
        'crossing': [],
        'others': []
    }

    # for all images
    for it, in_predicted in enumerate(args.in_predicted):
        print('Processing subject {} out of {}'.format(it+1, len(args.in_predicted)))
        predicted_im = nib.load(in_predicted)
        predicted_peaks = np.reshape(predicted_im.get_fdata(),
                                    np.append(predicted_im.shape[:3], (-1, 3)))

        # 1st. Classify voxels.
        predicted_labels = classify_peaks(predicted_peaks)

        # 2nd. For each voxel class, count TP rate.
        tag = os.path.basename(in_predicted)
        results['tag'].append(tag)

        for label, ind in CLASSES.items():
            prediction_match = np.logical_and(expected_labels == ind, predicted_labels == ind)
            nb_hits = np.count_nonzero(prediction_match)
            if nb_hits > 0:
                rate = float(nb_hits) / float(np.count_nonzero(expected_labels == ind))
            else:
                rate = 0.0
            results[label].append(rate)

    table = pd.DataFrame(results)
    table.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
