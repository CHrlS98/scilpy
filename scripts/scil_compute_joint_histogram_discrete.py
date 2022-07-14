#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from scilpy.io.image import get_data_as_mask

import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image_x')
    p.add_argument('in_image_y')
    p.add_argument('in_mask')
    p.add_argument('out_csv')

    return p


def create_joint_hist_csv(data_x, data_y, mask, out_csv):
    data_x = data_x[mask > 0]
    data_y = data_y[mask > 0]

    nbins_x = np.max(data_x) + 1  # if max value is 6, 7 bins (includes 0)
    nbins_y = np.max(data_y) + 1

    hist = np.zeros((nbins_x, nbins_y))
    for i in range(nbins_x):
        bins = np.bincount(data_y[data_x == i])
        hist[i, :len(bins)] = bins

    xlabels = np.arange(nbins_x)
    ylabels = np.arange(nbins_y)

    df = pd.DataFrame(hist, xlabels, ylabels)
    df.to_csv(out_csv)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image_x = nib.load(args.in_image_x).get_fdata().astype(np.uint8)
    image_y = nib.load(args.in_image_y).get_fdata().astype(np.uint8)
    mask = get_data_as_mask(nib.load(args.in_mask))

    create_joint_hist_csv(image_x, image_y, mask, args.out_csv)


if __name__ == '__main__':
    main()
