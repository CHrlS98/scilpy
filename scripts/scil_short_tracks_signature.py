#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from numba import njit
import nibabel as nib

from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)
from dipy.tracking.streamlinespeed import set_number_of_points
from scilpy.reconst.ftd import key_to_vox_index
from dipy.data import get_sphere
from dipy.direction import peak_directions
from scilpy.tractanalysis.todi_util import get_dir_to_sphere_id


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram')
    p.add_argument('in_vox2tracks')

    p.add_argument('out_vox2clusters')
    p.add_argument('out_peaks')

    p.add_argument('--num_points', type=int, default=4,
                   help='Number of points for resampling short-tracks.'
                        ' [%(default)s]')
    p.add_argument('--angle', type=float, default=20.0,
                   help='Minimum separation angle for extracting TOD'
                        ' peak directions. [%(default)s]')
    p.add_argument('--rel_th', default=0.1, type=float,
                   help='Relative threshold for peak extraction. '
                        '[%(default)s]')

    add_json_args(p)
    add_overwrite_arg(p)
    return p


@njit(cache=True)
def linalg_norm(arr, axis):
    norm = np.sqrt(np.sum(arr**2, axis=axis))
    return norm


@njit(cache=True)
def fingerprint_streamlines(streamlines, main_directions):
    signatures = []
    for s in streamlines:
        vecs = s[1:] - s[:-1]  # (num_points, 3)
        norm = linalg_norm(vecs, axis=-1)
        vecs = vecs[norm > 0] / norm[norm > 0].reshape((-1, 1))

        s_fingerprint = []
        for v in vecs:
            v = np.ascontiguousarray(v).reshape((1, 3))

            # on calcule la distance a chaque centroide
            cos_angles = main_directions.dot(v.T)
            closest_cluster = np.argmax(np.abs(cos_angles))
            fprint = closest_cluster + 1
            if cos_angles[closest_cluster] < 0:
                fprint *= -1.0
            s_fingerprint.append(fprint)
        signatures.append(s_fingerprint)
    return np.array(signatures)


@njit(cache=True)
def contains(arr, val):
    for a in arr:
        if np.all(a == val):
            return True
    return False


@njit(cache=True)
def merge_fingerprints(fingerprints, max_unique_fprints):
    merged_signatures = []

    unique_clusters = np.zeros((max_unique_fprints,
                                len(fingerprints[0])))
    num_clusters = 0

    for sign in fingerprints:
        flip_sign = -sign[::-1]
        if num_clusters > 0:
            if contains(unique_clusters[:num_clusters], sign):
                merged_signatures.append(sign)
            elif contains(unique_clusters[:num_clusters], flip_sign):
                merged_signatures.append(flip_sign)
            else:
                unique_clusters[num_clusters] = sign
                num_clusters += 1
                merged_signatures.append(sign)
        else:
            unique_clusters[num_clusters] = sign
            num_clusters += 1
            merged_signatures.append(sign)
    return merged_signatures


def compute_tod_peaks(resampled_strl, sphere, relative_th=0.1, min_angle=20):
    directions = resampled_strl[:, 1:] - resampled_strl[:, :-1]
    directions = np.reshape(directions, (-1, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    # symmetrize directions
    directions = np.append(directions, -directions, axis=0)

    # (N, 3) input
    sph_ind = get_dir_to_sphere_id(directions, sphere.vertices)
    bincount = np.bincount(sph_ind)
    odf = np.zeros(len(sphere.vertices))
    odf[:len(bincount)] = bincount

    peaks, _, _ = peak_directions(odf, sphere, relative_th, min_angle)
    return peaks.reshape((-1, 3)).astype(np.float32)


def write_peaks_to_volume(volume, peaks, vox):
    if len(peaks) > volume.shape[-2]:
        zeros = np.zeros(np.append(volume.shape[:3],
                                   peaks.shape))
        zeros[:, :, :, :volume.shape[-2]] = volume
        volume = zeros
    vox_idx = key_to_vox_index(vox)
    volume[vox_idx[0], vox_idx[1], vox_idx[2], :len(peaks)] = peaks
    return volume


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_vox2tracks])
    assert_outputs_exist(parser, args, [args.out_vox2clusters, args.out_peaks])

    lazy_tractogram = nib.streamlines.load(args.in_tractogram, lazy_load=True)
    print("Resampling tractogram...")
    resampled_strl = np.array([set_number_of_points(s, args.num_points)
                               for s in lazy_tractogram.streamlines],
                              order='C', dtype=np.float32)
    vox2tracks = json.load(open(args.in_vox2tracks))
    print("Loaded vox2tracks dictionary!", resampled_strl.shape)

    # sphere used to build TOD
    sphere = get_sphere('repulsion200')

    vox2clusters = {}
    cluster_peaks = np.zeros(np.append(lazy_tractogram.header['dimensions'],
                                       [5, 3]))
    for vox, strl in vox2tracks.items():
        sub_tracks = np.array([resampled_strl[idx] for idx in strl])

        centroids = compute_tod_peaks(sub_tracks, sphere,
                                      args.rel_th, args.angle)
        cluster_peaks = write_peaks_to_volume(cluster_peaks, centroids, vox)

        fingerprints = fingerprint_streamlines(sub_tracks, centroids)
        max_num_fingerprints = len(np.unique(fingerprints, axis=0))
        fingerprints = merge_fingerprints(fingerprints, max_num_fingerprints)

        clusters, cluster_counts = np.unique(np.asarray(fingerprints),
                                             return_counts=True, axis=0)
        # reorder clusters from most to least populous
        clusters = clusters[np.argsort(cluster_counts)[::-1]]
        cluster_ids = np.zeros((len(sub_tracks), ), dtype=int)
        for i, fprint in enumerate(fingerprints):
            cluster_ids[i] = np.argwhere(
                np.all(fprint == clusters, axis=-1)).squeeze()
        vox2clusters[vox] = cluster_ids.tolist()

    # if there are more than 5 peaks, won't show in MI-Brain
    cluster_peaks = np.reshape(cluster_peaks, cluster_peaks.shape[:3] + (-1,))\
        .astype(np.float32)

    # save outputs
    out_json = open(args.out_vox2clusters, 'w')
    json.dump(vox2clusters, out_json,
              indent=args.indent,
              sort_keys=args.sort_keys)
    nib.save(nib.Nifti1Image(cluster_peaks,
                             lazy_tractogram.header['voxel_to_rasmm']),
             args.out_peaks)
    print('DONE')


if __name__ == '__main__':
    main()
