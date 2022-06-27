#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from dipy.segment.clustering import QuickBundlesX
from dipy.tracking.metrics import length
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_endpoints',
                   help='Interface mask.')
    p.add_argument('in_wm',
                   help='Input white matter mask.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram,
                                 args.in_endpoints])

    endpoints_img = nib.load(args.in_endpoints)
    wm_img = nib.load(args.in_wm)
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    endpoints = get_data_as_mask(endpoints_img)
    wm_mask = get_data_as_mask(wm_img)

    # voxel space with origin corner for binning endpoints
    sft.to_vox()
    sft.to_corner()




if __name__ == '__main__':
    main()
