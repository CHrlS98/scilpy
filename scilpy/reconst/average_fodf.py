# -*- coding: utf-8 -*-

import logging
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh


def compute_avg_fodf(data, sphere, sh_order=8, input_sh_basis='descoteaux07',
                     output_sh_basis='descoteaux07_full'):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """

    # Safety checks

    # Computing average of fODFs
    # Lets first consider the case where we only consider  
    # the 8 immediate neighbors in the average
    padding = np.full(len(data.shape), 2)
    padding[-1] = 0
    augm_datashape = tuple(np.array(data.shape) + padding)
    augm_data = np.zeros(augm_datashape)
    augm_data[1:-1,1:-1,1:-1] = data
    avg_data = augm_data

    # add translated data to avg data for all neighbors

    # crop avg_data and return
    return 
