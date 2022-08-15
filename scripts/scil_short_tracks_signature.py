#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from numba import njit
import nibabel as nib

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args, add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tracking.tools import resample_streamlines_num_points
from dipy.tracking.streamlinespeed import set_number_of_points


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram')
    p.add_argument('in_vox2tracks')

    p.add_argument('out_vox2clusters')

    p.add_argument('--num_points', type=int, default=4,
                   help='Number of points for resampling short-tracks.'
                        ' [%(default)s]')
    p.add_argument('--angle', type=float, default=40.0,
                   help='Maximum deviation angle for estimating TOD'
                        ' peak directions. [%(default)s]')
    add_reference_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)
    return p


@njit(cache=True)
def linalg_norm(arr, axis):
    norm = np.sqrt(np.sum(arr**2, axis=axis))
    return norm


@njit(cache=True)
def approx_tod_peaks(streamlines, angle_th, max_centroids=10):
    centroids = np.zeros((max_centroids, 3), dtype=np.float32)
    num_centroids = 0

    for s in streamlines:
        vecs = s[1:] - s[:-1]  # (num_points, 3)
        norm = linalg_norm(vecs, axis=-1)
        vecs = vecs[norm > 0] / norm[norm > 0].reshape((-1, 1))

        # on passe a travers les vecteurs et on les ajoute au clusters
        for v in vecs:
            v = np.ascontiguousarray(v).reshape((1, 3))
            if num_centroids > 0:  # si on a déjà des clusters
                centroids_arr = centroids[:num_centroids].reshape((-1, 3))
                centroids_arr /= linalg_norm(centroids_arr,
                                             axis=-1).reshape((-1, 1))

                # on calcule la distance a chaque centroide
                cos_angles = centroids_arr.dot(v.T)
                closest_cluster = np.argmax(np.abs(cos_angles))
                angles = np.arccos(np.abs(cos_angles))

                # si on est assez proche d'un, on l'ajoute
                if angles[closest_cluster] < angle_th:
                    if cos_angles[closest_cluster] < 0:
                        v = -v
                    centroids[closest_cluster] += v.reshape((-1,))
                elif num_centroids < max_centroids:
                    # sinon on cree un cluster, si jamais on peut pas
                    # et qu'on est trop loin on ignore -> outlier.
                    centroids[num_centroids] = v.reshape((-1,))
                    num_centroids += 1
            else:  # si on a pas de clusters on en crée un
                centroids[num_centroids] = v.reshape((-1,))
                num_centroids += 1

    # TODO: Merge similar centroids.
    out_normalized = centroids[:num_centroids]
    out_normalized /= linalg_norm(out_normalized, axis=-1).reshape((-1, 1))

    out_merged = np.zeros_like(out_normalized)
    out_merged[0] = out_normalized[0]
    num_merged = 1
    for v in out_normalized[1:]:
        merged = np.reshape(out_merged[:num_merged], (-1, 3))
        v = np.ascontiguousarray(v).reshape((1, 3))
        cos_angles = merged.dot(v.T)
        closest_cluster = np.argmax(np.abs(cos_angles))
        angles = np.arccos(np.abs(cos_angles))

        if angles[closest_cluster] < angle_th:
            if cos_angles[closest_cluster] < 0:
                v = -v
            out_merged[closest_cluster] += v.reshape((-1,))
        else:
            out_merged[num_merged] = v.reshape((-1,))
            num_merged += 1

    out_merged = out_merged[:num_merged]
    out_merged /= linalg_norm(out_merged, axis=-1).reshape((-1, 1))
    return out_merged.reshape((-1, 3))


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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_vox2tracks])
    assert_outputs_exist(parser, args, args.out_vox2clusters)

    lazy_tractogram = nib.streamlines.load(args.in_tractogram, lazy_load=True)
    print("Resampling tractogram...")
    resampled_strl = np.array([set_number_of_points(s, args.num_points)
                               for s in lazy_tractogram.streamlines],
                              order='C', dtype=np.float32)
    vox2tracks = json.load(open(args.in_vox2tracks))
    print("Loaded vox2tracks dictionary!", resampled_strl.shape)

    # angle threshold in radians
    angle_th = np.deg2rad(args.angle)

    vox2clusters = {}
    for vox, strl in vox2tracks.items():
        sub_tracks = np.array([resampled_strl[idx] for idx in strl])

        centroids = approx_tod_peaks(sub_tracks, angle_th)
        # print('A')
        fingerprints = fingerprint_streamlines(sub_tracks, centroids)
        # print('B')
        max_num_fingerprints = len(np.unique(fingerprints, axis=0))

        fingerprints = merge_fingerprints(fingerprints, max_num_fingerprints)
        # print('C')
        clusters = np.unique(np.asarray(fingerprints), axis=0)
        cluster_ids = np.zeros((len(sub_tracks), ), dtype=int)
        for i, fprint in enumerate(fingerprints):
            cluster_ids[i] = np.argwhere(
                np.all(fprint == clusters, axis=-1)).squeeze()
        vox2clusters[vox] = cluster_ids.tolist()

    print('DONE')

    # save outputs
    out_json = open(args.out_vox2clusters, 'w')
    json.dump(vox2clusters, out_json,
              indent=args.indent,
              sort_keys=args.sort_keys)


if __name__ == '__main__':
    main()
