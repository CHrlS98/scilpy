#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np
from scilpy.reconst.ftd import FTDFit
from fury import actor, window
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere

from scilpy.reconst.utils import get_sh_order_and_fullness


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('ftd_json')
    p.add_argument('odf')

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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ftd = FTDFit.from_json(args.ftd_json)
    odf_image = nib.load(args.odf)
    odf = odf_image.get_fdata()
    visualize_as_lines(ftd)

    norm_max = 1.4
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
                if np.linalg.norm(dir) < norm_max:
                    centers.extend([p, p])
                    dirs.extend([-dir, dir])

    norms = np.linalg.norm(dirs, axis=-1)

    colors = np.abs(dirs / norms[..., None])
    centers = np.asarray(centers) - 0.5
    arrows = actor.arrow(centers, dirs, colors=colors,
                         heights=0.2*norms, tip_length=0.1)

    sphere = get_sphere('symmetric724')
    order, full_basis = get_sh_order_and_fullness(odf.shape[-1])
    b_mat = sh_to_sf_matrix(sphere, order,
                            full_basis=full_basis,
                            return_inv=False)
    odf_slicer = actor.odf_slicer(odf, sphere=sphere,
                                  B_matrix=b_mat, opacity=0.5)

    scene = window.Scene()
    scene.add(arrows)
    scene.add(odf_slicer)

    window.show(scene)


if __name__ == '__main__':
    main()
