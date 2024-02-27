# -*- coding: utf-8 -*-
import numpy as np
# from dipy.reconst.mapmri import MapmriModel, MapmriFit
from dipy.reconst.multi_voxel import MultiVoxelFit
import multiprocessing
import itertools

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")


def _fit_from_model_parallel(args):
    model = args[0]
    data = args[1]
    chunk_id = args[2]

    sub_fit_array = np.zeros((data.shape[0],), dtype='object')
    for i in range(data.shape[0]):
        if data[i].any():
            sub_fit_array[i] = model.fit(data[i])

    return chunk_id, sub_fit_array


def fit_from_model(model, data, mask=None, nbr_processes=None):
    """Fit the model to data

    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    data : np.ndarray (4d)
        Diffusion data.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    nbr_processes : int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    fit_array : np.ndarray
        Array containing the fit
    """
    data_shape = data.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(bool)
    else:
        mask_any = np.sum(data, axis=3).astype(bool)
        mask *= mask_any

    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes <= 0 \
        else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    masked_data_shape = (np.count_nonzero(mask), data_shape[3])

    data = data[mask].reshape(masked_data_shape)
    chunks = np.array_split(data, nbr_processes)

    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_fit_from_model_parallel,
                       zip(itertools.repeat(model),
                           chunks,
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    fit_array = np.zeros(data_shape[0:3], dtype='object')
    tmp_fit_array = np.zeros((np.count_nonzero(mask)), dtype='object')
    for i, fit in results:
        tmp_fit_array[chunk_len[i]:chunk_len[i+1]] = fit

    fit_array[mask] = tmp_fit_array
    fit_array = MultiVoxelFit(model, fit_array, mask)

    return fit_array

