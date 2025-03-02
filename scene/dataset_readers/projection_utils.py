
#
# Copyright (C) 2025, Jingwei Xu
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  xujw2023@shanghaitech.edu.cn,
#                        davidxujw@gmail.com
#


import numpy as np
from typing import Tuple

def getCullMaskPointCloudInFrame(
        h, w,
        xyz_homo: np.ndarray,    # [points, 4]
        w2c: np.ndarray,    # [4, 4]
        intr: np.ndarray,   # [3, 4]
) -> Tuple[np.ndarray, np.ndarray]:     # (mask=[points], valid_pix=[points, wh])
    camera_point = (w2c @ xyz_homo.T).T
    camera_point = camera_point / camera_point[..., 3][..., None]
    positive_z_mask = camera_point[..., 2] > 0

    pixel_point = (intr @ camera_point.T).T
    # [w, h] = [x, y]
    pix = pixel_point[..., :2] / pixel_point[..., 2][..., None]

    mask = (pix[..., 0] < w) * (pix[..., 0] > 0) * (pix[..., 1] < h) * (pix[..., 1] > 0) * positive_z_mask

    valid_pix = pix[mask].astype(np.int32)

    return mask, valid_pix

def getCertainSemanticMask(
        semantic_map: np.ndarray,
        screen_pix_coord: np.ndarray, # WH
        uncertain_check_range: int = 10,
):
    certain_mask = np.ones_like(screen_pix_coord[..., 0], dtype=bool)

    left_h = screen_pix_coord[..., 1] - uncertain_check_range
    right_h = screen_pix_coord[..., 1] + uncertain_check_range
    left_w = screen_pix_coord[..., 0] - uncertain_check_range
    right_w = screen_pix_coord[..., 0] + uncertain_check_range

    h, w = semantic_map.shape[0], semantic_map.shape[1]

    left_h_valid_mask = left_h > 0
    right_h_valid_mask = right_h < h
    left_w_valid_mask = left_w > 0
    right_w_valid_mask = right_w < w

    # 1
    left_h_left_w_valid_mask = left_h_valid_mask * left_w_valid_mask
    left_h_left_w_semantic = semantic_map[left_h[left_h_left_w_valid_mask], left_w[left_h_left_w_valid_mask]]
    origin_semantic = semantic_map[
        screen_pix_coord[..., 1][left_h_left_w_valid_mask], screen_pix_coord[..., 0][left_h_left_w_valid_mask]
    ]

    uncertain_mask = (left_h_left_w_semantic != origin_semantic)
    temp = certain_mask[left_h_left_w_valid_mask]
    temp[uncertain_mask] = False
    certain_mask[left_h_left_w_valid_mask] = temp

    # 2
    left_h_right_w_valid_mask = left_h_valid_mask * right_w_valid_mask
    left_h_right_w_semantic = semantic_map[left_h[left_h_right_w_valid_mask], right_w[left_h_right_w_valid_mask]]
    origin_semantic = semantic_map[
        screen_pix_coord[..., 1][left_h_right_w_valid_mask], screen_pix_coord[..., 0][left_h_right_w_valid_mask]
    ]

    uncertain_mask = (left_h_right_w_semantic != origin_semantic)
    temp = certain_mask[left_h_right_w_valid_mask]
    temp[uncertain_mask] = False
    certain_mask[left_h_right_w_valid_mask] = temp

    # 3
    right_h_left_w_valid_mask = right_h_valid_mask * left_w_valid_mask
    right_h_left_w_semantic = semantic_map[right_h[right_h_left_w_valid_mask], left_w[right_h_left_w_valid_mask]]
    origin_semantic = semantic_map[
        screen_pix_coord[..., 1][right_h_left_w_valid_mask], screen_pix_coord[..., 0][right_h_left_w_valid_mask]
    ]

    uncertain_mask = (right_h_left_w_semantic != origin_semantic)
    temp = certain_mask[right_h_left_w_valid_mask]
    temp[uncertain_mask] = False
    certain_mask[right_h_left_w_valid_mask] = temp

    # 4
    right_h_right_w_valid_mask = right_h_valid_mask * right_w_valid_mask
    right_h_right_w_semantic = semantic_map[right_h[right_h_right_w_valid_mask], left_w[right_h_right_w_valid_mask]]
    origin_semantic = semantic_map[
        screen_pix_coord[..., 1][right_h_right_w_valid_mask], screen_pix_coord[..., 0][right_h_right_w_valid_mask]
    ]

    uncertain_mask = (right_h_right_w_semantic != origin_semantic)
    temp = certain_mask[right_h_right_w_valid_mask]
    temp[uncertain_mask] = False
    certain_mask[right_h_right_w_valid_mask] = temp

    return certain_mask
