
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

import torch
import numpy as np
import cv2

def dilate_mask(mask, dilate_factor=5):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )

    mask = torch.from_numpy(mask).cuda().bool()

    return mask
