#!/usr/bin/env python3
import argparse
import numpy as np
from scilpy.reconst.ftd import FTDFit
from fury import actor, window


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('ftd_json')

    return p


def visualize_as_lines(ftd, n_per_edge=2):
    edge_ticks = np.linspace(start=0.0, stop=1.0,
                             num=n_per_edge,
                             endpoint=False)
    seeds = []
    for x in edge_ticks:
        for y in edge_ticks:
            for z in edge_ticks:
                seeds.append([x, y, z])
    seeds = np.asarray(seeds)
    print(seeds)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ftd = FTDFit.from_json(args.ftd_json)
    visualize_as_lines(ftd)
    1/0

    ftd_grid_pos = np.indices((5, 5, 1), dtype=float)
    ftd_grid_pos = ftd_grid_pos.reshape(3, -1).T / 5.0 +\
        np.array([0.0, 0.0, 0.5])
    centers = []
    dirs = []
    for vox_pos in ftd.defined_voxels():
        if vox_pos[2] != 1:
            continue  # just visualize the central slice

        positions = np.asarray(vox_pos, dtype=float) + ftd_grid_pos
        for p in positions:
            for dir in ftd[p]:
                centers.extend([p]*2)
                dirs.extend([dir, -dir])

    norms = np.linalg.norm(dirs, axis=-1)
    colors = np.abs(dirs / norms[..., None])
    arrows = actor.arrow(centers, dirs, colors=colors, heights=0.2)

    scene = window.Scene()
    scene.add(arrows)
    window.show(scene)


if __name__ == '__main__':
    main()
