# -*- coding:utf-8 -*-
import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
from scipy.special import eval_legendre
from scipy.ndimage import convolve, gaussian_filter1d
from tqdm import tqdm

def _exp(wfilter_x, wfilter_y, wfilter_z):
    return np.exp(-(wfilter_x**2 + wfilter_y**2 + wfilter_z**2))


def _G4(vertices, wfilter_x, wfilter_y, wfilter_z):
    x_prime = np.reshape(vertices[:, 0], (-1, 1, 1, 1)) * wfilter_x[None, ...] + \
              np.reshape(vertices[:, 1], (-1, 1, 1, 1)) * wfilter_y[None, ...] + \
              np.reshape(vertices[:, 2], (-1, 1, 1, 1)) * wfilter_z[None, ...]
    g4 = (16.0*x_prime**4 - 48*x_prime**2 + 12) * _exp(wfilter_x, wfilter_y, wfilter_z)

    # Normalize to ensure discrete filter sums to 0 over spatial dimensions
    g4 = g4 - np.mean(g4, axis=(1, 2, 3), keepdims=True)
    # Normalize by L2 norm
    l2_norm = np.sqrt(np.sum(g4**2, axis=(1, 2, 3), keepdims=True))
    g4 = g4 / l2_norm
    return g4


def _H4(vertices, wfilter_x, wfilter_y, wfilter_z):
    x_prime = np.reshape(vertices[:, 0], (-1, 1, 1, 1)) * wfilter_x[None, ...] + \
              np.reshape(vertices[:, 1], (-1, 1, 1, 1)) * wfilter_y[None, ...] + \
              np.reshape(vertices[:, 2], (-1, 1, 1, 1)) * wfilter_z[None, ...]
    h4 = (5.10495*x_prime**5 - 38.29678*x_prime**3 + 36.70429*x_prime) * _exp(wfilter_x, wfilter_y, wfilter_z)

    # Normalize to ensure discrete filter sums to 0 over spatial dimensions
    h4 = h4 - np.mean(h4, axis=(1, 2, 3), keepdims=True)
    # Normalize by L2 norm
    l2_norm = np.sqrt(np.sum(h4**2, axis=(1, 2, 3), keepdims=True))
    h4 = h4 / l2_norm
    return h4


def compute_micro_odf(data, sphere, window_halfwidth, sh_order_max, sh_basis_type, legacy):
    """Compute micro ODF from image data using spherical harmonics.
    
    Parameters
    ----------
    data : ndarray
        Input image data.
    sphere : Sphere
        Sphere object containing vertices for filter directions.
    window_halfwidth : float
        Half width of the filtering window.
    sh_order_max : int
        Maximum order of spherical harmonics.
    sh_basis_type : str
        Type of spherical harmonics basis.
    legacy : bool
        Whether to use legacy basis convention.
    
    Returns
    -------
    sh_coeffs : ndarray
        Spherical harmonics coefficients of the ODF.
    """
    # Parameters for estimating ODF from microscopy image
    sampling_delta = 3.0 / (window_halfwidth * np.sqrt(2))
    print(f'Sampling delta: {sampling_delta}\nWindow_halfwidth: {window_halfwidth}')
    wfilter_x, wfilter_y, wfilter_z = \
        np.meshgrid(*[np.arange(-window_halfwidth, window_halfwidth+1)*sampling_delta for _ in range(3)], indexing='ij')

    # FRT definition
    n_coeffs = int((sh_order_max + 2) * (sh_order_max + 1) / 2)
    _, l = sph_harm_ind_list(sh_order_max)
    FRT = np.diag(2.0*np.pi*eval_legendre(l, 0))
    B, B_inv = sh_to_sf_matrix(sphere, sh_order_max=sh_order_max,
                               basis_type=sh_basis_type, smooth=0.0,
                               legacy=legacy, return_inv=True)
    n_dirs = len(sphere.vertices)

    # Estimate filters
    G4 = _G4(sphere.vertices, wfilter_x, wfilter_y, wfilter_z)
    H4 = _H4(sphere.vertices, wfilter_x, wfilter_y, wfilter_z)

    # Compute ODF coefficients
    sh_coeffs = np.zeros(data.shape + (n_coeffs,))
    sf_max = np.zeros(data.shape)
    for i in tqdm(range(n_dirs)):
        # Convolve image with G4 and H4 filters
        conv_g4 = convolve(data, G4[i], mode='nearest') * sampling_delta**3
        conv_h4 = convolve(data, H4[i], mode='nearest') * sampling_delta**3

        # Quadrature filter pair so we sum the squares of the responses
        sf_i = conv_g4**2 + conv_h4**2
        b_i = np.expand_dims(B[:, i], axis=(0, 1, 2))
        # print(sf_i.shape, b_i.shape)
        sh_coeffs += sf_i[..., None] * b_i
        sf_max = np.maximum(sf_max, sf_i)

    # FRT to align ODF peak with underlying fiber direction
    sh_coeffs = sh_coeffs.dot(FRT)
    return sh_coeffs


def compute_micro_odf_from_gradient(data, sphere, sigma, sh_order_max, sh_basis, legacy):
    """Compute micro ODF from image data using spherical harmonics.
    
    Parameters
    ----------
    data : ndarray
        Input image data.
    sphere : Sphere
        Sphere object containing vertices for filter directions.
    sigma : float
        Standard deviation of the Gaussian smoothing kernel.
    sh_order_max : int
        Maximum order of spherical harmonics.
    sh_basis_type : str
        Type of spherical harmonics basis.
    legacy : bool
        Whether to use legacy basis convention.
    
    Returns
    -------
    sh_coeffs : ndarray
        Spherical harmonics coefficients of the ODF.
    """
    G4_x = gaussian_filter1d(data, sigma, axis=0, mode='nearest', order=1)
    G4_y = gaussian_filter1d(data, sigma, axis=1, mode='nearest', order=1)
    G4_z = gaussian_filter1d(data, sigma, axis=2, mode='nearest', order=1)
    G4_grad = np.stack((G4_x, G4_y, G4_z), axis=-1)

    n_coeffs = int((sh_order_max + 2) * (sh_order_max + 1) / 2)
    _, l = sph_harm_ind_list(sh_order_max)
    FRT = np.diag(2.0*np.pi*eval_legendre(l, 0))
    B, _ = sh_to_sf_matrix(sphere, sh_order_max=sh_order_max,
                               basis_type=sh_basis, smooth=0.0,
                               legacy=legacy, return_inv=True)
    n_dirs = len(sphere.vertices)

    sh_coeffs = np.zeros(data.shape + (n_coeffs,))
    for i in tqdm(range(n_dirs)):
        v = sphere.vertices[i].reshape(3, 1)
        sf_i = G4_grad.dot(v)**2
        b_i = np.expand_dims(B[:, i], axis=(0, 1, 2))
        sh_coeffs += sf_i * b_i

    sh_coeffs = sh_coeffs.dot(FRT)
    print(sh_coeffs.max())
    return sh_coeffs