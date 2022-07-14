#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from scilpy.io.image import get_data_as_mask

import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_image')
    p.add_argument('in_mask')
    p.add_argument('out_csv')

    return p


def create_hist_csv(data, mask, out_csv):
    bins = np.bincount(data[mask > 0])
    labels = np.arange(len(bins))
    d = {'labels': labels,
         'count': bins}
    df = pd.DataFrame(d)
    df.to_csv(out_csv, index=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_image).get_fdata().astype(np.uint8)
    mask = get_data_as_mask(nib.load(args.in_mask))

    create_hist_csv(image, mask, args.out_csv)


if __name__ == '__main__':
    main()
