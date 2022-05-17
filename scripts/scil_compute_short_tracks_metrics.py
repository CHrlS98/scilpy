#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')
    p.add_argument('out_tractogram', help='Output tractogram file (trk).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram])
    assert_outputs_exist(parser, args, [args.out_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    data_per_streamline = []
    orientation_mat = np.eye(3)
    for streamline in sft.streamlines:
        orientation = np.reshape(streamline[-1] - streamline[0], (3, 1))
        if (orientation_mat.dot(orientation).sum() <
           orientation_mat.dot(-orientation).sum()):
            orientation = -orientation
        orientation = orientation / np.linalg.norm(orientation)
        data_per_streamline.append(orientation.squeeze())

    data_per_streamline = np.asarray(data_per_streamline)
    sft.data_per_streamline['orient'] = data_per_streamline

    color = data_per_streamline
    color += 1.0
    color /= 2.0
    color = (color * 255).astype(int)
    tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                   (len(sft.streamlines[i]), 1))
           for i in range(len(sft.streamlines))]
    sft.data_per_point['color'] = tmp
    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
