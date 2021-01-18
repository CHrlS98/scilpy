# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask

EPS = 0.13587
SH_CONST = 0.5 / np.sqrt(np.pi)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_vf', help='Input volume fractions file.')
    p.add_argument('out_wm_mask', help='Output WM mask.')

    p.add_argument('--th', type=float, default=2.3)

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_vf)
    assert_outputs_exist(parser, args, args.out_wm_mask)

    vf_img = nib.load(args.in_vf)
    vf = vf_img.get_fdata()

    pCSF = vf[..., 0]
    pWM_union_pGM = np.sum(vf[..., 1:], axis=-1)

    brain_mask = pCSF + pWM_union_pGM > 0.

    wm_mask = np.zeros_like(brain_mask)
    wm_mask[brain_mask] =\
        pWM_union_pGM[brain_mask] / pCSF[brain_mask] > args.th

    nib.save(nib.Nifti1Image(wm_mask.astype(np.uint8), vf_img.affine),
             args.out_wm_mask)


if __name__ == '__main__':
    main()
