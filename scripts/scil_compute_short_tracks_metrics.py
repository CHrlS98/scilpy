#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import compress_streamlines
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.metrics import length
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)

from fury import window, actor


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


def resample(strl, step_size):
    resampled_strl = [strl[0]]
    print(resampled_strl)
    print(strl)
    n_index = 0  # indice of next point on original streamline

    # fonctionne mais il manque la fin de la streamline.
    can_continue = True
    on_last_segment = False
    while can_continue:
        if n_index == len(strl) - 2:
            on_last_segment = True

        if not on_last_segment:
            while np.sum((resampled_strl[-1] - strl[n_index + 1])**2) < step_size:
                n_index += 1
        else:  # on_last_segment
            can_continue = np.sum((strl[-1] - resampled_strl[-1])**2) > step_size

        # le prochain point se trouve sur le segment courant
        v = strl[n_index+1] - strl[n_index]
        x0 = strl[n_index]
        p = resampled_strl[-1]
        # quadratic equation coefficients
        a = v[0]**2 + v[1]**2 + v[2]**2
        b = 2 * (v[0] * x0[0] + v[1] * x0[1] + v[2] * x0[2] -
                 p[0] * v[0] - p[1] * v[1] - p[2] * v[2])
        c = x0[0]**2 + x0[1]**2 + x0[2]**2 - 2 * p[0] * x0[0]\
            - 2 * p[1] * x0[1] - 2 * p[2] * x0[2]\
            + p[0]**2 + p[1]**2 + p[2]**2 - step_size**2
        # solve quadratic equation
        t = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        resampled_strl.append(x0 + t * v)

    length_resampled = length(resampled_strl, True)
    print(length_resampled[1:] - length_resampled[:-1])
    return resampled_strl


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    strl_i = 4
    step = 2.0

    strl = sft.streamlines[strl_i:strl_i+1]

    compressed = compress_streamlines(strl, 0.1)
    compressed_sft = StatefulTractogram.from_sft(compressed, sft)

    resampled = [resample(compressed[0], step)]
    resampled_scilpy = resample_streamlines_step_size(compressed_sft, step)
    strl_resampled_scilpy = resampled_scilpy.streamlines[0:1]
    length_resampled = length(strl_resampled_scilpy[0], True)
    print(length_resampled[1:] - length_resampled[:-1])

    resampled_scilpy_a = actor.line(strl_resampled_scilpy, colors=(1, 1, 0))
    resampled_scilpy_dots = actor.dots(np.asarray(strl_resampled_scilpy[0]),
                                       color=(1, 1, 0))

    resampled_a = actor.line(resampled, colors=(1, 0, 0))
    resampled_dots = actor.dots(np.asarray(resampled[0]))

    # compressed_a = actor.line(compressed, colors=(0, 1, 0), opacity=0.5)
    s = window.Scene()
    s.add(resampled_scilpy_a)
    s.add(resampled_scilpy_dots)
    # s.add(compressed_a)
    s.add(resampled_a)
    s.add(resampled_dots)
    # s.add(line_a)
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
