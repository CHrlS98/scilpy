#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.io.utils import add_overwrite_arg, add_reference_arg, assert_inputs_exist, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram', help='Input tractogram file (trk).')

    add_overwrite_arg(p)
    add_reference_arg(p)
    return p


def hermite_interpolation(p0, p1, m0, m1):
    """
    Compute the Hermite interpolation between two points.
    """
    t = np.linspace(0, 1, 100).reshape((-1, 1))
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    interp = h00*p0 + h10*m0 + h01*p1 + h11*m1
    # print(interp.shape)
    return h00*p0 + h10*m0 + h01*p1 + h11*m1


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    # interp = []
    # for s in sft.streamlines:
    #     p0 = np.reshape(s[0], (1, 3))
    #     p1 = np.reshape(s[-1], (1, 3))
    #     m0 = np.reshape(s[1] - s[0], (1, 3))
    #     m1 = np.reshape(s[-1] - s[-2], (1, 3))
    #     m0 /= np.linalg.norm(m0)
    #     m1 /= np.linalg.norm(m1)
    #     interp.append(hermite_interpolation(p0, p1, m0, m1))

    compressed = compress_streamlines(sft.streamlines, 0.1)
    from fury import window, actor
    s = window.Scene()
    line_a = actor.line(sft.streamlines, opacity=0.5)
    interp_a = actor.line(compressed)
    s.add(interp_a)
    s.add(line_a)
    window.show(s)


def _main():
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
