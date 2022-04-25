# -*- coding: utf-8 -*-

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.reconst.bingham import bingham_to_peak_direction
from scilpy.tractanalysis.grid_intersections import grid_intersections


def lobe_specific_metric_map_along_streamlines(sft, bingham_coeffs,
                                               metric, max_theta,
                                               length_weighting):
    """
    Compute mean map for a given lobe-specific metric along streamlines.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    bingham_coeffs : ndarray
        Array of shape (X, Y, Z, N_LOBES, NB_PARAMS) containing
        the Bingham distributions parameters.
    metric : ndarray or list
        Array of shape (X, Y, Z, N_LOBES) containing the lobe-specific
        metric of interest.
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        Bingham peak direction.
    length_weighting : bool
        If True, will weigh the metric values according to segment lengths.
    """

    metric_sum_list, weights_list = \
        lobe_metric_sum_along_streamlines(sft, bingham_coeffs,
                                          metric, max_theta,
                                          length_weighting)

    for metric_sum, weights in zip(metric_sum_list, weights_list):
        non_zeros = np.nonzero(metric_sum)
        metric_sum[non_zeros] /= weights[non_zeros]

    return metric_sum_list


def lobe_metric_sum_along_streamlines(sft, bingham_coeffs, metrics,
                                      max_theta, length_weighting):
    """
    Compute a sum map along a bundle for a given lobe-specific metric.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines needed.
    bingham_coeffs : ndarray (X, Y, Z, N_LOBES, NB_PARAMS)
        Bingham distributions parameters volume.
    metric : ndarray (X, Y, Z, N_LOBES) or list
        The lobe-specific metric(s) of interest.
    max_theta : float
        Maximum angle in degrees between the fiber direction and the
        Bingham peak direction.
    length_weighting : bool
        If True, will weight the metric values according to segment lengths.

    Returns
    -------
    metric_sum_map : np.array
        Lobe-specific metric sum map.
    weight_map : np.array
        Segment lengths.
    """

    sft.to_vox()
    sft.to_corner()

    if isinstance(metrics, np.ndarray):
        metric_list = [metrics]
    else:
        metric_list = metrics

    metric_sum_map = [np.zeros(metric.shape[:-1]) for metric in metric_list]
    weight_map = [np.zeros(metric.shape[:-1]) for metric in metric_list]
    min_cos_theta = np.cos(np.radians(max_theta))

    all_crossed_indices = grid_intersections(sft.streamlines)
    for crossed_indices in all_crossed_indices:
        segments = crossed_indices[1:] - crossed_indices[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        # Those starting points are used for the segment vox_idx computations
        seg_start = crossed_indices[non_zero_lengths]
        vox_indices = (seg_start + (0.5 * segments)).astype(int)

        normalization_weights = np.ones_like(seg_lengths)
        if length_weighting:
            normalization_weights = seg_lengths

        normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

        for vox_idx, seg_dir, norm_weight in zip(vox_indices,
                                                 normalized_seg,
                                                 normalization_weights):
            vox_idx = tuple(vox_idx)
            bingham_at_idx = bingham_coeffs[vox_idx]  # (5, N_PARAMS)

            bingham_peak_dir = bingham_to_peak_direction(bingham_at_idx)
            cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                      bingham_peak_dir.T))

            metric_val = 0.0
            for i_metric in range(len(metric_list)):
                if (cos_theta > min_cos_theta).any():
                    lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)
                    metric_val = metric_list[i_metric][vox_idx][lobe_idx]

                metric_sum_map[i_metric][vox_idx] += metric_val * norm_weight
                weight_map[i_metric][vox_idx] += norm_weight

    return metric_sum_map, weight_map
