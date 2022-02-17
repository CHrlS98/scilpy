#!/usr/bin/env python3
import argparse
import nibabel as nib
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh, sh_to_sf_matrix
from dipy.sims.voxel import multi_tensor_odf
from dipy.core.sphere import Sphere

from fury import actor, window
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    # p.add_argument('in_sh')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sphere = get_sphere('repulsion724').subdivide(2)

    evals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(90, 0), (0, 0)]
    fractions = [50, 50]

    odf = multi_tensor_odf(sphere.vertices, evals, angles, fractions)
    sh = sf_to_sh(odf, sphere, sh_order=8)

    order, full_basis = get_sh_order_and_fullness(sh.shape[-1])

    sf = sh_to_sf(sh, sphere, sh_order=order, full_basis=full_basis)
    sf[sf < 0] = 0.0

    n_samples = 1000
    ind = []
    for i in range(n_samples):
        ind.append(sample_distribution(sf))

    dirs = sphere.vertices[ind, :]
    pi = np.random.uniform(-0.5, 0.5, size=(n_samples, 3))
    pf = pi + dirs
    lines = [*zip(pi, pf)]

    scene = window.Scene()
    lactor = actor.line(lines)
    odf_actor = actor.odf_slicer(odf[None, None, None, :], scale=1.0, sphere=sphere, opacity=0.5, colormap=(255, 255, 255))
    scene.add(lactor)
    scene.add(odf_actor)
    scene.add(actor.axes())
    window.show(scene)

    some_dir = np.array([[1.0, 0.0, 0.0]])
    costheta = np.abs(dirs.dot(some_dir.T))
    print(costheta.shape)
    hist, _ = np.histogram(costheta, bins=20)
    plt.bar(np.arange(len(hist)), hist)
    plt.show()


if __name__ == '__main__':
    main()
