#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import numpy as np
import nibabel as nib
from scipy.interpolate import splprep, splev
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_ind_list, sh_to_sf_matrix

from scilpy.io.utils import (add_sh_basis_args,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)

from fury import window, actor

DELTA_SPLINE = 0.001
DEFAULT_TRK_OUTPUT = 'sim.trk'
DEFAULT_FODF_OUTPUT = 'fodf.nii.gz'
DEFAULT_ASYM_FODF_OUTPUT = 'asym_fodf.nii.gz'
DEFAULT_WM_MASK_OUTPUT = 'wm.nii.gz'
DEFAULT_ENDPOINTS_OUTPUT = 'endpoints.nii.gz'


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_config', help='JSON configuration file.')

    p.add_argument('--out_tractogram', default=DEFAULT_TRK_OUTPUT,
                   help='Output tractogram file. [{}]'
                        .format(DEFAULT_TRK_OUTPUT))
    p.add_argument('--out_sym_fodf', default=DEFAULT_FODF_OUTPUT,
                   help='Output symmetric fODF file. [{}]'
                        .format(DEFAULT_FODF_OUTPUT))
    p.add_argument('--out_asym_fodf', default=DEFAULT_ASYM_FODF_OUTPUT,
                   help='Output asymmetric fODF file. [{}]'
                        .format(DEFAULT_ASYM_FODF_OUTPUT))
    p.add_argument('--out_wm_mask', default=DEFAULT_WM_MASK_OUTPUT,
                   help='Output white matter mask file. [{}]'
                        .format(DEFAULT_WM_MASK_OUTPUT))
    p.add_argument('--out_endpoints_mask', default=DEFAULT_ENDPOINTS_OUTPUT,
                   help='Output endpoint mask file. [{}]'
                        .format(DEFAULT_ENDPOINTS_OUTPUT))

    p.add_argument('--volume_size', nargs=3, default=[10, 10, 10], type=int,
                   help='Size of the 3-dimensional volume.')
    p.add_argument('--fiber_decay', default=10, type=float,
                   help='Sharpness for a single fiber lobe.')

    p.add_argument('--sh_order', default=8, type=int,
                   help='Maximum SH order for FODF reconstruction.')
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    return p


def transform_streamlines_to_vox(streamlines, volume_size):
    streamlines = copy.deepcopy(streamlines)
    # streamlines are in RASMM with origin corner
    # we need to find the translation such that the
    # most negative point goes to (0, 0, 0) and
    # a scaling such that the maximum value does not
    # exceed the volume dimensions
    min_coords = np.array([[0.0, 0.0, 0.0]])
    max_coords = np.array([[0.0, 0.0, 0.0]])
    for s in streamlines:
        # volume is (N, 3)
        min_pos = np.min(s, axis=0)
        subarr = np.array([min_pos, min_coords.squeeze()])
        min_coords = np.min(subarr, axis=0)

        max_pos = np.max(s, axis=0)
        subarr = np.array([max_pos, max_coords.squeeze()])
        max_coords = np.max(subarr, axis=0)

    # transform to voxel space
    max_coords -= min_coords
    # find the optimal scaling to apply to fill the volume dimensions.
    # The 0.01 offset ensures that no streamlines points falls outside the
    # bounding box.
    scaling = np.min(np.array(volume_size) / (max_coords + 0.01))
    for s in streamlines:
        s -= min_coords
        s *= scaling

    return streamlines


def streamlines_to_segments(streamlines):
    seg_dirs = []
    seg_lengths = []
    seg_pos = []
    for s in streamlines:
        segments = s[1:] - s[:-1]
        lengths = np.linalg.norm(segments, axis=-1, keepdims=True)
        dirs = segments / lengths

        seg_pos.append(s[:-1] + segments / 2)
        seg_dirs.append(dirs)
        seg_lengths.append(lengths)

    seg_dirs = np.vstack(seg_dirs)
    seg_lengths = np.vstack(seg_lengths)
    seg_pos = np.vstack(seg_pos)

    return seg_dirs, seg_lengths, seg_pos


def vonMisesFisher(x, mu, kappa, lmax_norm=False):
    ck = kappa / (4.0*np.pi*np.sinh(kappa))
    ret = ck * np.exp(kappa * mu.dot(x.T))
    if lmax_norm:
        return ret / ret.max()
    return ret


def generate_streamlines(config_dict, volume_size):
    streamlines = []
    for bundle in config_dict['bundles']:
        centroid = np.array(bundle['centroids']).astype(float)
        up = np.array(bundle['up_vectors']).astype(float)

        tck, u = splprep(centroid.T, s=0)
        centroid_approx = np.array(splev(u, tck)).T
        centroid_offset = np.array(splev(u + DELTA_SPLINE, tck)).T

        # frenet frame
        T = centroid_offset - centroid_approx
        T /= np.linalg.norm(T, axis=-1, keepdims=True)

        up /= np.linalg.norm(up, axis=-1, keepdims=True)
        N = up - np.array([up[i].dot(T[i]) for i in range(len(up))])\
            .reshape((-1, 1))*T

        B = np.cross(T, N)

        # now that we have our frame at each point we can generate streamlines
        max_radius = bundle['radius']
        n_streamlines = bundle['n_streamlines']
        for _ in range(n_streamlines):
            radius = np.sqrt(np.random.uniform()) * max_radius
            theta = np.random.uniform() * 2.0 * np.pi
            points = centroid_approx + radius * np.cos(theta) * N\
                + radius * np.sin(theta) * B
            tck, _ = splprep(points.T, s=0)
            x_approx = np.array(
                splev(np.linspace(0, 1, bundle['n_steps']), tck)).T
            streamlines.append(x_approx)

    # transform streamlines to voxel space
    streamlines = transform_streamlines_to_vox(streamlines, volume_size)
    strl_start = [s[0] for s in streamlines]
    strl_stop = [s[-1] for s in streamlines]
    endpoints_vox = np.floor(strl_start + strl_stop).astype(int)
    unique = np.unique(endpoints_vox, axis=0)
    endpoints = np.zeros(volume_size, dtype=bool)
    endpoints[unique[:, 0], unique[:, 1], unique[:, 2]] = True

    return streamlines, endpoints


def generate_fiber_odf(streamlines, volume_size, fiber_decay,
                       sh_order, sh_basis):
    # generate ground truth fODF by modeling fiber segments with
    # von Mises-Fisher distributions (similar to TODI).
    dirs, lengths, positions = streamlines_to_segments(streamlines)
    vox_ids = np.floor(positions).astype(int)
    unique_voxids = np.unique(vox_ids, axis=0)

    sphere = get_sphere('repulsion724').subdivide(2)
    _, B_inv = sh_to_sf_matrix(sphere, sh_order,
                               basis_type=sh_basis,
                               full_basis=True)
    odf = np.zeros(volume_size + [B_inv.shape[-1]])
    sf_max = 0.0
    for vox_id in unique_voxids:
        vox_mask = np.all(vox_ids == vox_id, axis=-1)
        vox_dirs = dirs[vox_mask]
        vox_lengths = lengths[vox_mask]
        vox_pos = positions[vox_mask]
        center_of_mass = np.mean(vox_pos, axis=0)
        sf = np.zeros((1, len(sphere.vertices)))
        for dir, leng, pos in zip(vox_dirs, vox_lengths, vox_pos):
            pos = pos - np.floor(pos)
            center_of_mass = center_of_mass - np.floor(center_of_mass)
            ctr_to_pos = pos - center_of_mass
            if ctr_to_pos.dot(dir) < 0:
                dir = -dir
            sf += vonMisesFisher(sphere.vertices, dir,
                                 fiber_decay,
                                 lmax_norm=True) * leng
            sf_max = max(sf.max(), sf_max)

        odf[vox_id[0], vox_id[1], vox_id[2]] = np.dot(sf, B_inv).squeeze()

    # heuristic so that the largest ODF is not more than 2 voxels wide
    # (this only matters for visualisation)
    return odf / sf_max


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_config)
    assert_outputs_exist(parser, args, [args.out_tractogram,
                                        args.out_sym_fodf,
                                        args.out_asym_fodf,
                                        args.out_endpoints_mask,
                                        args.out_wm_mask])

    with open(args.in_config, 'r') as f:
        config = json.load(f)

    streamlines, endpoints = generate_streamlines(config, args.volume_size)

    template_header = nib.Nifti1Header()
    template_header.set_data_shape(args.volume_size)
    template_header.set_sform(np.identity(4), code='scanner')

    sft_header = nib.Nifti1Header().from_header(template_header)

    sft = StatefulTractogram(streamlines, reference=sft_header,
                             space=Space.VOX,
                             origin=Origin.TRACKVIS)

    endpoints_header = nib.Nifti1Header().from_header(template_header)
    endpoints_header.set_data_dtype(np.dtype(np.uint8))
    nib.save(nib.Nifti1Image(endpoints.astype(np.uint8), None,
                             header=endpoints_header),
             args.out_endpoints_mask)

    save_tractogram(sft, args.out_tractogram)

    # generate ground truth fODF by modeling fiber segments with
    # von Mises-Fisher distributions (similar to TODI).
    fodf = generate_fiber_odf(streamlines, args.volume_size,
                              args.fiber_decay, args.sh_order,
                              args.sh_basis)

    fodf_header = nib.Nifti1Header().from_header(template_header)
    fodf_header.set_data_dtype(np.dtype(np.float32))
    nib.save(nib.Nifti1Image(fodf.astype(np.float32), None,
                             header=fodf_header),
             args.out_asym_fodf)

    _, orders = sph_harm_ind_list(args.sh_order, full_basis=True)
    nib.save(nib.Nifti1Image(fodf[..., orders % 2 == 0].astype(np.float32),
                             None, header=fodf_header),
             args.out_sym_fodf)

    # because the background is 0, the WM mask is the non-zero voxels in fodf
    wm_mask = np.any(np.abs(fodf) > 0.0, axis=-1)
    nib.save(nib.Nifti1Image(wm_mask.astype(np.uint8), None,
                             header=endpoints_header),
             args.out_wm_mask)


if __name__ == '__main__':
    main()
