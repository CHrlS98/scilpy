#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import argparse
import numpy as np
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_verbose_arg, assert_inputs_exist)
from scilpy.tractanalysis.grid_intersections import grid_intersections
from scipy.spatial import ConvexHull
from fury import window, actor
from itertools import combinations

from dipy.reconst.shm import sh_to_sf
from dipy.core.sphere import Sphere
from dipy.data import get_sphere


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument('in_tractogram', help='Input tractogram file.')
    p.add_argument('in_fodf', help='Input fodf file.')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_fodf])

    fodf_img = nib.load(args.in_fodf)
    fodf = fodf_img.get_fdata()

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    streamlines = sft.streamlines
    seeds = sft.data_per_streamline['seeds']

    endpoints = []
    for s in streamlines:
        endpoints.append(s[0])
        endpoints.append(s[-1])

    cvx_hull = ConvexHull(endpoints)
    outer_endpoints = np.asarray(endpoints)[cvx_hull.vertices].tolist()
    tentative_paths = ArraySequence(
        np.asarray(list(combinations(outer_endpoints, 2)))
        .astype(np.float32))

    norm = sh_to_sf(fodf, get_sphere('symmetric362'), sh_order=8)
    norm = np.linalg.norm(norm, axis=-1)
    fodf[norm > 0] /= norm[norm > 0][..., None]

    # compute probability of each tentative path
    vox_coordinates = grid_intersections(tentative_paths)
    probabilities = []
    for intersections, line in zip(vox_coordinates, tentative_paths):
        intersections = (intersections + 0.5).astype(int)
        indices = np.unique(intersections, axis=0)
        dir = line[1] - line[0]
        dir /= np.linalg.norm(dir)
        afd = sh_to_sf(fodf[indices[:, 0], indices[:, 1], indices[:, 2]],
                       Sphere(xyz=dir.reshape((1, 3))), sh_order=8)
        probability = np.mean(afd.astype(np.float128))
        probabilities.append(probability)

    probabilities = np.asarray(probabilities).astype(np.float32)
    # print(np.mean(probabilities))

    threshold = np.mean(probabilities)
    tentative_paths = tentative_paths[probabilities > threshold]
    colors = probabilities[probabilities > threshold] / np.max(probabilities)
    colors = np.broadcast_to(colors[..., None], (colors.shape[0], 3))

    strl = actor.line(tentative_paths, colors=colors, opacity=0.7)
    seeds_actr = actor.dots(seeds, color=(1, 1, 1))
    pts = actor.point(outer_endpoints, colors=(1, 0, 0))
    scene = window.Scene()
    scene.add(pts)
    scene.add(strl)
    # scene.add(seeds_actr)
    window.show(scene)


if __name__ == "__main__":
    main()
