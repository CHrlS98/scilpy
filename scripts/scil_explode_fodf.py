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

from numba import jit


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--in_sh',
                   default='fibercup/reconstructions/fodf_8/fodf_8_descoteaux.nii.gz')
    return p


def _filter_fast(sh, B, B_inv):
    shape = sh.shape
    out_sh = np.zeros((shape[0]*2, shape[1]*2, shape[2]*2, shape[3]))
    sh = np.pad(sh, ((1, 1), (1, 1), (1, 1), (0, 0)))
    for id_x in range(shape[0]):
        for id_y in range(shape[1]):
            for id_z in range(shape[2]):
                win = sh[id_x:id_x+3, id_y:id_y+3, id_z:id_z+3]
                win_sf = win.dot(B)

                # negative values are ringing artifacts
                win_sf[win_sf < 0] = 0.0
                win_sf_zero = win_sf.copy()
                win_sf_zero[1, 1, 1] = 0.0

                # Sums inside 8 sub-regions
                filters = np.zeros((2, 2, 2, win_sf.shape[-1]))
                for sub_i in range(2):
                    for sub_j in range(2):
                        for sub_k in range(2):
                            filters[sub_i, sub_j, sub_k] = \
                                win_sf_zero[sub_i:sub_i+2,
                                            sub_j:sub_j+2,
                                            sub_k:sub_k+2].sum(axis=(0, 1, 2))
                print(filters.max())
                vsum = np.sum(filters, axis=(0, 1, 2))
                mask = vsum > 0.0
                filters[:, :, :, np.nonzero(mask)[0]] /= vsum[mask]

                filtered_sf = filters * win_sf[1, 1, 1]
                filtered_sh = filtered_sf.dot(B_inv)
                out_sh[id_x*2:id_x*2+2,
                       id_y*2:id_y*2+2,
                       id_z*2:id_z*2+2] = filtered_sh

    return out_sh


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_sh)
    sh = image.get_fdata()
    sphere = get_sphere('repulsion724')
    B, B_inv = sh_to_sf_matrix(sphere, sh_order=8)

    out_sh = _filter_fast(sh, B, B_inv)
    nib.save(nib.Nifti1Image(out_sh, image.affine), 'out_sh.nii.gz')


if __name__ == '__main__':
    main()
