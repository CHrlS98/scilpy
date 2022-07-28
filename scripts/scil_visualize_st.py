#!/usr/bin/env python3
import argparse
import json
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg

from fury import actor, window, colormap


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_st')
    p.add_argument('in_vox2tracks')
    p.add_argument('in_vox2ids')

    p.add_argument('vox_id', nargs=3, type=int)

    add_reference_arg(p)
    return p


def snapshot_voxel(streamlines, vox2tracks, vox2ids, vox_id):
    key = np.array2string(np.asarray(vox_id))
    strl_ids = vox2tracks[key]
    cluster_ids = vox2ids[key]
    nb_clusters = np.max(cluster_ids) + 1
    strl = streamlines[strl_ids]
    colors = colormap.distinguishable_colormap(nb_colors=np.max(cluster_ids)+1)

    actors = []
    for cluster_i in range(nb_clusters):
        c_strl = strl[np.asarray(cluster_ids) == cluster_i]
        opacity = float(len(c_strl)) / float(len(strl))
        color = np.tile(colors[cluster_i], len(c_strl)).reshape(-1, 3)
        a = actor.line(c_strl, colors=color, opacity=opacity)
        actors.append(a)

    s = window.Scene()
    s.add(*actors)
    window.show(s)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vox2tracks = json.load(open(args.in_vox2tracks, 'r'))
    vox2ids = json.load(open(args.in_vox2ids, 'r'))

    sft = load_tractogram_with_reference(parser, args, args.in_st)
    snapshot_voxel(sft.streamlines, vox2tracks, vox2ids, args.vox_id)


if __name__ == '__main__':
    main()
