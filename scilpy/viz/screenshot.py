# -*- coding: utf-8 -*-

from fury import window, actor, colormap
import numpy as np


def display_slices(volume_actor, slices,
                   output_filename, axis_name,
                   view_position, focal_point,
                   peaks_actor=None, streamlines_actor=None):
    # Setting for the slice of interest
    if axis_name == 'sagittal':
        volume_actor.display(slices[0], None, None)
        if peaks_actor:
            peaks_actor.display(slices[0], None, None)
        view_up_vector = (0, 0, 1)
    elif axis_name == 'coronal':
        volume_actor.display(None, slices[1], None)
        if peaks_actor:
            peaks_actor.display(None, slices[1], None)
        view_up_vector = (0, 0, 1)
    else:
        volume_actor.display(None, None, slices[2])
        if peaks_actor:
            peaks_actor.display(None, None, slices[2])
        view_up_vector = (0, 1, 0)

    # Generate the scene, set the camera and take the snapshot
    ren = window.Renderer()
    ren.add(volume_actor)
    if streamlines_actor:
        ren.add(streamlines_actor)
    elif peaks_actor:
        ren.add(peaks_actor)
    ren.set_camera(position=view_position,
                   view_up=view_up_vector,
                   focal_point=focal_point)

    window.snapshot(ren, size=(1920, 1080), offscreen=True,
                    fname=output_filename)


def prepare_scene(axis_name, shape):
    """
    Prepare scene and camera for visualizing slice in axis_name orientation
    """
    if axis_name == 'sagittal':
        view_position = [-280.0,
                         (shape[1] - 1) / 2.0,
                         (shape[2] - 1) / 2.0]
        view_center = [(shape[0] - 1) / 2.0,
                       (shape[1] - 1) / 2.0,
                       (shape[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
        zoom_factor = 2.0 / shape[1] if shape[1] > shape[2] else 2.0 / shape[2]
    elif axis_name == 'coronal':
        view_position = [(shape[0] - 1) / 2.0,
                         280.0,
                         (shape[2] - 1) / 2.0]
        view_center = [(shape[0] - 1) / 2.0,
                       (shape[1] - 1) / 2.0,
                       (shape[2] - 1) / 2.0]
        view_up = [0.0, 0.0, 1.0]
        zoom_factor = 2.0 / shape[0] if shape[0] > shape[2] else 2.0 / shape[2]
    elif axis_name == 'axial':
        view_position = [(shape[0] - 1) / 2.0,
                         (shape[1] - 1) / 2.0,
                         -280.0]
        view_center = [(shape[0] - 1) / 2.0,
                       (shape[1] - 1) / 2.0,
                       (shape[2] - 1) / 2.0]
        view_up = [0.0, 1.0, 0.0]
        zoom_factor = 2.0 / shape[0] if shape[0] > shape[1] else 2.0 / shape[1]

    scene = window.Scene()
    scene.projection('parallel')
    print('view_pos: ', view_position)
    print('view_center: ', view_center)
    print('view_up: ', view_up)
    scene.set_camera(position=view_position,
                     focal_point=view_center,
                     view_up=view_up)
    scene.zoom(zoom_factor)

    return scene


def crop_data_along_axis(data, idx, axis_name):
    """
    Crop data along a dimension specified by axis name at index idx
    """
    if axis_name == 'sagittal':
        return data[idx:idx+1, :, :]
    elif axis_name == 'coronal':
        return data[:, idx:idx+1, :]
    elif axis_name == 'axial':
        return data[:, :, idx:idx+1]


def display_scene(actors, shape, window_size, orientation,
                  interactor, output, title):
    """
    Prepare and display a scene containing 'actors'
    """
    scene = prepare_scene(orientation, shape)
    for actor in actors:
        scene.add(actor)

    showm = window.ShowManager(scene, size=window_size,
                               title=title,
                               reset_camera=False,
                               interactor_style=interactor)
    showm.initialize()
    showm.start()

    if output:
        window.snapshot(scene, fname=output, size=window_size)


def get_translation_matrix(translation):
    return np.array([[1.0, 0.0, 0.0, translation[0]],
                     [0.0, 1.0, 0.0, translation[1]],
                     [0.0, 0.0, 1.0, translation[2]],
                     [0.0, 0.0, 0.0, 1.0]])


def prepare_texture_slicer_actor(data, min_value, max_value,
                                 axis_name, colormap_lut=None):
    value_range = [data.min(), data.max()]
    if min_value is not None:
        value_range[0] = min_value
    if max_value is not None:
        value_range[1] = max_value
    value_range = tuple(value_range)

    if axis_name == 'sagittal':
        slicer_actor =\
            actor.slicer(data,
                         affine=get_translation_matrix((1.0, 0.0, 0.0)),
                         value_range=value_range, interpolation='nearest',
                         lookup_colormap=colormap_lut)
        slicer_actor.display_extent(0, 0, 0, data.shape[1] - 1,
                                    0, data.shape[2] - 1)
    elif axis_name == 'coronal':
        slicer_actor =\
             actor.slicer(data,
                          affine=get_translation_matrix((0.0, -1.0, 0.0)),
                          value_range=value_range, interpolation='nearest',
                          lookup_colormap=colormap_lut)
        slicer_actor.display_extent(0, data.shape[0] - 1, 0, 0,
                                    0, data.shape[2] - 1)
    elif axis_name == 'axial':
        slicer_actor =\
            actor.slicer(data,
                         affine=get_translation_matrix((0.0, 0.0, 1.0)),
                         value_range=value_range, interpolation='nearest',
                         lookup_colormap=colormap_lut)
        slicer_actor.display_extent(0, data.shape[0] - 1, 0,
                                    data.shape[1] - 1, 0, 0)

    return slicer_actor
