# -*- coding: utf-8 -*-
import itertools
import logging
import os

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import length
from nibabel.streamlines import ArraySequence

from scilpy.io.hdf5 import construct_hdf5_group_from_streamlines
from scilpy.io.streamlines import save_tractogram
from scilpy.tractanalysis.bundle_operations import remove_outliers_qb
from scilpy.tractograms.streamline_and_mask_operations import \
    compute_streamline_segment
from scilpy.tractograms.streamline_operations import \
    (remove_loops as perform_remove_loops,
     remove_loops_and_sharp_turns, remove_shap_turns_qb)


def extract_longest_segments_from_profile(strl_indices, atlas_data):
    """
    For one given streamline, find the labels at both ends.

    Parameters
    ----------
    strl_indices: np.ndarray
        The indices of all voxels traversed by this streamline.
    atlas_data: np.ndarray
        The loaded image containing the labels.

    Returns
    -------
    segments_info: list[dict]
        A list of length 1 with the information dict if , else, an empty list.
    """
    # toDo. background/wm is defined as label 0 in segmenting func, but should
    #  be asked to user.

    start_label = None
    end_label = None
    start_idx = None
    end_idx = None

    nb_underlying_voxels = len(strl_indices)

    # Find the starting point.
    # Advancing if we start in a non-interesting position (label 0, background
    # or WM). Start_label will be the first GM region encountered
    # (corresponding to a label).
    current_vox = 0
    while start_label is None and current_vox < nb_underlying_voxels:
        if atlas_data[tuple(strl_indices[current_vox])] > 0:
            start_label = atlas_data[tuple(strl_indices[current_vox])]
            start_idx = current_vox
        current_vox += 1

    if start_label is None:
        return []

    # Continuing to advance along the streamline. If we do not find a label 0
    # somewhere (WM), this is a weird streamline never leaving GM. Returning []
    found_wm = False
    while not found_wm and current_vox < nb_underlying_voxels:
        if atlas_data[tuple(strl_indices[current_vox])] == 0:
            found_wm = True
        current_vox += 1
    if current_vox >= nb_underlying_voxels or not found_wm:
        return []

    # Find the ending point. As before, moving back as long as we are in a non-
    # interesting position.
    current_vox = nb_underlying_voxels - 1
    while end_label is None and current_vox > start_idx:
        if atlas_data[tuple(strl_indices[current_vox])] > 0:
            end_label = atlas_data[tuple(strl_indices[current_vox])]
            end_idx = current_vox
        current_vox -= 1

    if end_label is None or end_idx <= start_idx + 1:
        return []

    return [{'start_label': start_label,
             'start_index': start_idx,
             'end_label': end_label,
             'end_index': end_idx}]


def compute_connectivity(indices, atlas_data, real_labels, segmenting_func):
    """
    Segments a tractogram into "bundles", or "connections" between all pairs
    of labels.

    Parameters
    ----------
    indices: ArraySequence
        The list of 3D indices [i, j, k] of all voxels traversed by all
        streamlines. This is the output of our uncompress function.
    atlas_data: np.ndarray
        The loaded image containing the labels.
    real_labels: np.ndarray
        The list of labels of interest in the image.
    segmenting_func: Callable
        The function used for segmentation.
        Ex: extract_longest_segments_from_profile

    Returns
    -------
    connectivity: dict
        A dict containing one key per real_labels (ex, 1, 2) (starting point).

        - The value of connectivity[1] is again a dict with again the
            real_labels as keys.

        - The value of connectivity[1][2] is a list of length n, where n is
            the number of streamlines starting in 1 and finishing in 2. Each
            value is a dict of the following shape:

           >>> 'strl_idx': int --> The idex of the streamline in the raw data.
           >>> 'in_idx:    int -->
           >>> 'out_idx': int  -->
    """
    connectivity = {k: {lab: [] for lab in real_labels} for k in real_labels}

    # toDo. real_labels is not used in segmenting func!
    for strl_idx, strl_vox_indices in enumerate(indices):
        # Managing streamlines out of bound.
        if (np.array(strl_vox_indices) > atlas_data.shape).any():
            continue

        # Finding start_label and end_label.
        segments_info = segmenting_func(strl_vox_indices, atlas_data)
        for si in segments_info:
            connectivity[si['start_label']][si['end_label']].append(
                {'strl_idx': strl_idx,
                 'in_idx': si['start_index'],
                 'out_idx': si['end_index']})

    return connectivity


def construct_hdf5_from_connectivity(
        whole_sft, vox_sizes, indices, points_to_idx,
        real_labels, con_info, hdf5_file, saving_options, out_paths,
        prune_from_length, min_length, max_length,  # step 1
        remove_loops, loop_max_angle,               # step 2
        remove_outliers, outlier_threshold,         # step 3
        remove_curv_dev, curv_qb_distance,          # step 4
        nbr_cpu
):
    """
    Parameters
    ----------
    whole_sft: StatefulTractogram
        The tractogram.
    vox_sizes: list
        The 3D voxel size.
    indices: ArraySequence
        Results from uncompress.
    points_to_idx: ArraySequence
        Results from uncompress.
    real_labels: np.ndarray
        The labels.
    con_info: dict
        The result from compute_connectivity.
    hdf5_file: hdf5 file
        The opened hdf5_file to which to add the bundles (as groups).
    saving_options: dict
        Steps for which intermediate files should be saved on disk (not in the
        hdf5). Keys are: 'raw', 'intermediate', 'discarded', 'final'. Values
        are True or False.
    out_paths: dict
        Name of the intermediate files. Keys are: 'raw', 'invalid_length',
        'valid_length', 'loops', 'outliers', 'qb_curv', 'no_loops', 'inliers',
        'final'. They will be saved if not stated otherwise in saving_options.
    prune_from_length: bool
        If true, limit length between [min_length, max_length]. Else, skip
        pruning (step 1).
    min_length: float
    max_length: float
    remove_loops: bool
        If true, remove looping streamlines. Else skip step 2.
    loop_max_angle: float
    remove_outliers: bool
        If true, remove outliers using Quickbundles. Else skip step 3.
    outlier_threshold: float
    remove_curv_dev: bool
        If true, remove sharp turns base on Quickbundles. Else skip step 4.
    curv_qb_distance: float
    nbr_cpu: int
        Number of cpu for steps allowing multiprocessing.
    """
    whole_sft.to_vox()
    whole_sft.to_corner()

    comb_list = list(itertools.combinations(real_labels, r=2))
    comb_list.extend(zip(real_labels, real_labels))

    # Each connection is processed independently. Multiprocessing would be
    # a burden on the I/O of most SSD/HD.
    iteration_counter = 0
    for in_label, out_label in comb_list:
        iteration_counter += 1
        if iteration_counter > 0 and iteration_counter % 100 == 0:
            logging.info('Processing connection {}/{}'
                         .format(iteration_counter, len(comb_list)))
        logging.debug('Processing connection {}/{}: {} - {}'
                      .format(iteration_counter, len(comb_list),
                              in_label, out_label))

        # Extracting this connection's info from the big dict con_info
        pair_info = []
        if out_label in con_info[in_label]:
            pair_info.extend(con_info[in_label][out_label])
        if in_label in con_info[out_label]:
            pair_info.extend(con_info[out_label][in_label])
        if len(pair_info) == 0:
            logging.debug("No streamlines found for this connection: not "
                          "saving in the hdf5.")
            continue

        # Preparing streamlines. Keeping only the segment between the two
        # associated labels.
        logging.debug("- Keeping only the segments between the two associated "
                      "labels for each streamline. Any data_per_point will be "
                      "lost.")
        current_streamlines = []
        connecting_ids = []
        for connection in pair_info:
            strl_idx = connection['strl_idx']
            curr_streamlines = compute_streamline_segment(
                whole_sft.streamlines[strl_idx],
                indices[strl_idx],
                connection['in_idx'],
                connection['out_idx'],
                points_to_idx[strl_idx])
            current_streamlines.append(curr_streamlines)
            connecting_ids.append(strl_idx)
        raw_dps = whole_sft.data_per_streamline[connecting_ids]
        current_sft = StatefulTractogram.from_sft(current_streamlines, whole_sft,
                                                  data_per_streamline=raw_dps,
                                                  data_per_point={})
        _save_intermediate(current_sft, saving_options, out_paths,
                           in_label, out_label,
                           save_type='raw', step_name='raw')
        del current_streamlines

        # Cleaning.
        # Each step is processed from the previous 'success'
        #   1. raw         -> length pass/fail
        #   2. length pass -> loops pass/fail
        #   3. loops pass  -> outlier detection pass/fail
        #   4. outlier detection pass -> qb curvature pass/fail
        #   5. qb curvature pass == final connections

        # STEP 1
        if prune_from_length:
            logging.debug("- Step 1: Pruning by length: [{}, {}]"
                          .format(min_length, max_length))
            valid_length_ids, invalid_length_ids = _prune_segments(
                current_sft.streamlines, min_length, max_length, vox_sizes[0])

            # Discarded:
            discarded_sft = current_sft[invalid_length_ids]
            _save_intermediate(discarded_sft, saving_options, out_paths,
                               in_label, out_label, save_type='discarded',
                               step_name='invalid_length')

            # Remaining:
            current_sft = current_sft[valid_length_ids]
            _save_intermediate(current_sft, saving_options, out_paths,
                               in_label, out_label, save_type='intermediate',
                               step_name='valid_length')
        else:
            logging.debug("- Step 1 skipped (no pruning from length)")

        if len(current_sft) == 0:
            logging.debug("- No remaining streamlines. Stopping now.")
            continue

        # STEP 2
        if remove_loops:
            logging.debug("- Step 2: Removing loops > {}"
                          .format(loop_max_angle))
            no_loop_ids, _ = perform_remove_loops(
                current_sft.streamlines, loop_max_angle, num_processes=nbr_cpu)
            loop_ids = np.setdiff1d(np.arange(len(current_sft)), no_loop_ids)

            # Discarded:
            discarded_sft = current_sft[loop_ids]
            _save_intermediate(discarded_sft, saving_options, out_paths,
                               in_label, out_label, save_type='discarded',
                               step_name='loops')

            # Remaining:
            no_loops_sft = current_sft[no_loop_ids]
            _save_intermediate(no_loops_sft, saving_options, out_paths,
                               in_label, out_label, save_type='intermediate',
                               step_name='no_loops')
        else:
            logging.debug("- Step 2 skipped (not removing loops)")

        if len(current_sft) == 0:
            logging.debug("- No remaining streamlines. Stopping now.")
            continue

        # STEP 3
        if remove_outliers:
            logging.debug("- Step 3: Removing outliers (Qb threshold: {})."
                          .format(outlier_threshold))
            outliers_ids, inliers_ids = remove_outliers_qb(
                current_sft.streamlines, outlier_threshold, nb_samplings=10,
                fast_approx=True)

            # Discarded:
            discarded_sft = current_sft[outliers_ids]
            _save_intermediate(discarded_sft, saving_options, out_paths,
                               in_label, out_label,  save_type='discarded',
                               step_name='outliers')

            # Remaining:
            current_sft = current_sft[inliers_ids]
            _save_intermediate(current_sft, saving_options, out_paths,
                               in_label, out_label, save_type='intermediate',
                               step_name='inliers')
        else:
            logging.debug("- Step 3 skipped (not removing outliers)")

        if len(current_sft) == 0:
            logging.debug("- No remaining streamlines. Stopping now.")
            continue

        # STEP 4
        if remove_curv_dev:
            logging.debug("- Step 4: Removing sharp turns (Qb threshold: {})"
                          .format(curv_qb_distance))
            no_qb_curv_ids = remove_shap_turns_qb(
                current_sft.streamlines, qb_threshold=curv_qb_distance)
            qb_curv_ids = np.setdiff1d(np.arange(len(current_sft)),
                                       no_qb_curv_ids)

            # Discarded:
            discarded_sft = current_sft[qb_curv_ids]
            _save_intermediate(discarded_sft, saving_options, out_paths,
                               in_label, out_label,  save_type='discarded',
                               step_name='qb_curv')

            # Remaining:
            current_sft = current_sft[no_qb_curv_ids]
            # (Saving below; they are the final streamlines, saved even if
            # step 4 not done.)
        else:
            logging.debug("- Step 4 skipped (not removing sharp turns)")

        # Final streamlines.
        # Due to the cutting, streamlines can become invalid.
        # toDo DEMANDER A FRANCOIS CA FAIT QUOI CA
        indices = []
        for i in range(len(current_sft)):
            norm = np.linalg.norm(
                np.gradient(current_sft.streamlines[i], axis=0), axis=1)
            if (norm < 0.001).any():  # or len(sft.streamlines[i]) <= 1:
                indices.append(i)
        indices = np.setdiff1d(range(len(current_sft)),
                               indices).astype(np.uint32)
        current_sft = current_sft[indices]

        _save_intermediate(current_sft, saving_options, out_paths,
                           in_label, out_label, save_type='final',
                           step_name='final')

        # Saving final streamlines in the hdf5
        group = hdf5_file.create_group('{}_{}'.format(in_label, out_label))
        construct_hdf5_group_from_streamlines(
            group, current_sft.streamlines,
            dps=current_sft.data_per_streamline)


def _prune_segments(segments, min_length, max_length, vox_size):
    # A REMPLACER PAR FONCTION OFFICIELLE DANS TRACTOGRAMME.
    lengths = list(length(segments) * vox_size)
    valid = []
    invalid = []

    for i, tuple_zip in enumerate(zip(segments, lengths)):
        _, le = tuple_zip
        if min_length <= le <= max_length:
            valid.append(i)
        else:
            invalid.append(i)
    return valid, invalid


def _save_intermediate(sft, saving_options, out_paths, in_label, out_label,
                       save_type, step_name):
    if saving_options[save_type]:
        out_name = os.path.join(out_paths[step_name],
                                '{}_{}.trk'.format(in_label, out_label))
        save_tractogram(sft, out_name, no_empty=True)