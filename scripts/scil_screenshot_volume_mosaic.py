#!/usr/bin/env python3
import argparse
from itertools import product
import nibabel as nib
import numpy as np
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, snapshot)
from scilpy.viz.scene_utils import (create_texture_slicer, create_scene,
                                    render_scene)

from fury import window
from fury.colormap import create_colormap
import vtk

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, NoNorm
from matplotlib.cm import ScalarMappable


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_volume')

    add_overwrite_arg(p)
    return p


def create_cmap_lookup(vmin, vmax, name):
    # create lookup table
    lut = vtk.vtkLookupTable()
    v = np.arange(vmin, vmax)  # number of colors excluding 0
    cmap = create_colormap(v, name=name)
    cmap = np.vstack(([[0.0, 0.0, 0.0]], cmap))  # set 0 to black

    lut.SetNumberOfTableValues(len(cmap))
    lut.SetTableRange(0, len(cmap))
    for i, c in enumerate(cmap):
        lut.SetTableValue(i, c[0], c[1], c[2], 1.0)
    lut.Build()

    return cmap, lut


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_volume)

    image = nib.load(args.in_volume).get_fdata()

    n_rows = 2
    n_cols = 5
    n_imgs = n_rows * n_cols
    fig, axes = plt.subplots(n_rows, n_cols)

    orientation = 'coronal'
    indices = np.linspace(0, image.shape[1] - 1, n_imgs + 2).astype(int)
    indices = indices[1:-1]  # removes the black slices at both ends of volume

    colors, lut = create_cmap_lookup(0, image.max(), 'jet')

    for i, pos in enumerate(product(range(n_rows), range(n_cols))):
        idx = indices[i]
        texture = create_texture_slicer(image, orientation, slice_index=idx,
                                        cmap_lut=lut)
        scene = create_scene([texture], orientation, idx, image.shape)
        im = snapshot(scene, None)
        im = im[40:-40, 40:-40]  # crop
        axes[pos[0], pos[1]].imshow(im)
        axes[pos[0], pos[1]].axis('off')

    cbar = fig.colorbar(ScalarMappable(cmap=ListedColormap(colors)), ax=axes,
                        orientation='horizontal', values=np.arange(len(colors)))
    plt.show()


if __name__ == '__main__':
    main()
