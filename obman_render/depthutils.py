import numpy as np
import cv2


def convert_depth(tmp_depth):
    """
    Prepares for saving by putting infinity depth to 0 and scaling
    valid depth values between 1 and 255
    """
    depth3channels = cv2.imread(tmp_depth, flags=3)
    if depth3channels is None:
        raise ValueError('Could not read image {}'.format(tmp_depth))
    depth = depth3channels[:, :, 0]
    dead_pixels = depth == 1e10
    depth_max = depth[~dead_pixels].max()
    depth_min = depth[~dead_pixels].min()
    scaled_depth = 254 * (depth - depth_min) / (depth_max - depth_min) + 1
    scaled_depth[dead_pixels] = 0
    return scaled_depth, depth_max, depth_min

def convert_depth_and_swap_pixed_intensity(tmp_depth):
    """
    Prepares for saving by putting infinity depth to 0 and scaling
    valid depth values between 1 and 255. This function is almost the same as convert_depth(tmp_depth), 
    but is swaps the pixel pixel intensity values, so that closer objects appear lighter 
    and objects that are further away appear darker. 
    """
    depth3channels = cv2.imread(tmp_depth, flags=3)
    if depth3channels is None:
        raise ValueError('Could not read image {}'.format(tmp_depth))
    depth = depth3channels[:, :, 0]
    dead_pixels = depth == 1e10
    depth_max = depth[~dead_pixels].max()
    depth_min = depth[~dead_pixels].min()
    # Using (depth - depth_max) instead of (depth - depth_min):
    scaled_depth = 254 * (abs(depth - depth_max)) / (depth_max - depth_min) + 1
    scaled_depth[dead_pixels] = 0
    return scaled_depth, depth_max, depth_min

def read_depth(depth_path, depth_min, depth_max):
    """!!! Untested"""
    depth = cv2.imread(depth_path)
    assert depth.max(
    ) == 255, 'Max value of depth jpg should be 255, not {}'.format(
        depth.max())
    scaled_depth = (depth - 1) / 254 * (depth_max - depth_min) + depth_min
    return scaled_depth
