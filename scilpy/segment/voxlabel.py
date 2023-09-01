# -*- coding: utf-8 -*-
import numpy as np


def _classify_two_directions(peaks, cos_tol, nufid, labels):
    #  identify two-direction voxels (straight and bending)
    idx,idy, idz = np.nonzero(nufid == 2)
    for ind in zip(idx, idy, idz):
        p0 = peaks[ind][0]
        p1 = peaks[ind][1]
        dot = p0.dot(p1)
        if dot < cos_tol:
            labels[ind] = 2
        else:
            labels[ind] = 2.5


def _classify_four_directions(peaks, cos_tol, nufid, labels):
    #  identify two-direction voxels (straight and bending)
    idx,idy, idz = np.nonzero(nufid == 4)
    for ind in zip(idx, idy, idz):
        p = peaks[ind][:4]
        sym = True
        for pi in p:
            dots = pi.dot(p.T)  # (1, 4)
            if np.min(dots) >= cos_tol:
                labels[ind] = 4.5
                sym = False
                break
        if sym:
            labels[ind] = 4


def classify_peaks_asym(peaks, bend_tol):
    peak_norms = np.linalg.norm(peaks, axis=-1)
    nufid = np.count_nonzero(peak_norms, axis=-1)
    labels = np.zeros_like(nufid, dtype=float)

    # normalize peaks for later
    peaks[peak_norms > 0] /= peak_norms[peak_norms > 0][..., None]

    # label "obvious" configurations
    labels[nufid == 1] = 1
    labels[nufid == 3] = 3
    labels[nufid > 4] = 5  # others

    cos_tol = np.cos(np.deg2rad(180 - bend_tol))
    _classify_two_directions(peaks, cos_tol, nufid, labels)
    _classify_four_directions(peaks, cos_tol, nufid, labels)
    return labels
