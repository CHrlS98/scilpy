#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docstring for scilpy.cli.scil_fodf_apply_priors
"""
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from dipy.reconst.shm import sh_to_sf, sf_to_sh, sh_to_sf_matrix, order_from_ncoef, sph_harm_ind_list
from dipy.data import get_sphere
from scilpy.io.utils import assert_headers_compatible,\
                            assert_inputs_exist,\
                            assert_outputs_exist,\
                            add_overwrite_arg,\
                            add_sh_basis_args,\
                            parse_sh_basis_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf',
                   help='Input FODF image onto which priors should be applied.')
    p.add_argument('in_priors',
                   help='Input SH priors to apply.')
    p.add_argument('in_mask',
                   help='Mask of region of interest to enhance with priors.')
    p.add_argument('out_efod', help='Output enhanced FODF.')
    p.add_argument('--out_priors', help='Optional output of SH priors inside mask.')
    p.add_argument('--kappa', type=float, default=5.0,
                   help='Concentration parameter for direction smoothing [%(default)s].')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Sigma for spatial smoothing [%(default)s].')

    add_sh_basis_args(p)
    add_overwrite_arg(p)
    return p


def smooth_sf(sf, sphere, kappa=5.0):
    """
    Smooth SF using von Mises-Fisher distribution

    :param sf: Description
    :param kappa: Description
    """ 
    weights = sphere.vertices.dot(sphere.vertices.T)
    weights = np.exp(kappa*weights)
    weights /= np.sum(weights, axis=0)
    sf_smooth = sf.dot(weights)
    return sf_smooth


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_fodf = nib.load(args.in_fodf)
    in_priors = nib.load(args.in_priors)
    in_mask = nib.load(args.in_mask)

    required_inputs = [args.in_fodf, args.in_priors, args.in_mask]
    assert_inputs_exist(parser, required_inputs)
    assert_outputs_exist(parser, args, args.out_efod, args.out_priors)
    assert_headers_compatible(parser, required_inputs)

    fodf = in_fodf.get_fdata()
    priors = in_priors.get_fdata()
    mask = in_mask.get_fdata().astype(bool)
    priors_in_mask = priors[mask]

    sh_basis_fodf, legacy_fodf = parse_sh_basis_arg(args)
    sh_order_fodf = order_from_ncoef(fodf.shape[-1])

    sh_basis_priors, legacy_priors = parse_sh_basis_arg(args)
    sh_order_priors = order_from_ncoef(priors.shape[-1])
    sphere = get_sphere(name='repulsion724')
    output_priors = np.zeros(priors.shape)

    priors_sf = sh_to_sf(priors_in_mask, sphere,sh_order_max=sh_order_priors,
                         basis_type=sh_basis_priors, legacy=legacy_priors)
    priors_sf = smooth_sf(priors_sf, sphere, kappa=args.kappa)
    smooth_priors_sh = sf_to_sh(priors_sf, sphere, sh_order_max=sh_order_priors,
                                basis_type=sh_basis_priors, legacy=legacy_priors)
    output_priors[mask] = smooth_priors_sh

    # normalized convolution for spatial smoothing
    output_priors[mask] = gaussian_filter(output_priors*mask.astype(float)[..., None],
                                       sigma=1.0, axes=(0,1,2))[mask]
    norm_weights = gaussian_filter(mask.astype(float), sigma=1.0)
    output_priors[mask] /= norm_weights[mask][..., None]

    if args.out_priors:
        nib.save(nib.Nifti1Image(output_priors, in_priors.affine), args.out_priors)

    # Transform priors back to sf (again)
    smooth_priors_sf = sh_to_sf(output_priors[mask], sphere,
                                sh_order_max=sh_order_priors,
                                basis_type=sh_basis_priors,
                                legacy=legacy_priors)

    # L-max normalization for applying to FODF
    smooth_priors_sf /= np.max(smooth_priors_sf, keepdims=True)
    fodf_sf = sh_to_sf(fodf[mask], sphere, sh_order_max=sh_order_fodf,
                       basis_type=sh_basis_fodf, legacy=legacy_fodf)
    efod_sf = fodf_sf * smooth_priors_sf
    efod_sf = efod_sf / np.max(efod_sf, axis=1, keepdims=True)\
        * np.max(fodf_sf, axis=1, keepdims=True)

    # Back to SH (last time)
    efod_sh = sf_to_sh(efod_sf, sphere=sphere, sh_order_max=sh_order_fodf,
                       basis_type=sh_basis_fodf, legacy=legacy_fodf)
    output_efod = fodf
    output_efod[mask] = efod_sh

    nib.save(nib.Nifti1Image(output_efod, in_fodf.affine), args.out_efod)


if __name__ == '__main__':
    main()
