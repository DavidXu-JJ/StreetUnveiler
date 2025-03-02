
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
import torch

semantic_color = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0],
    [128, 0, 255],
    [0, 128, 128],
    [128, 128, 0],
    [255, 192, 203],
    [0, 128, 0],
    [128, 128, 128],
    [192, 192, 192],
    [128, 0, 0],
    [0, 0, 128],
    [0, 128, 0],
    [128, 0, 128],
    [128, 128, 0]
], dtype=np.uint8)

semantic_color_torch = torch.from_numpy(semantic_color)
if torch.cuda.is_available():
    semantic_color_torch = semantic_color_torch.to(torch.cuda.current_device())

#---------------- Cityscapes semantic segmentation
cityscapes_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
cityscapes_classes_ind_map = {cn: i for i, cn in enumerate(cityscapes_classes)}

cityscapes_dynamic_classes = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

cityscapes_human_classes = [
    'person', 'rider'
]

waymo_classes_in_cityscapes = {
    'unknwon': ['train'],
    'Vehicle': ['car', 'truck', 'bus'],
    'Pedestrian': ['person'],
    'Sign': ['traffic light', 'traffic sign'],
    'Cyclist': ['rider', 'motorcycle', 'bicycle']
}

concerned_classes = {
    'road': ['road'],
    'sidewalk': ['sidewalk'],
    'building': ['building', 'wall', 'fence'],
    'vegetation': ['vegetation'],
    'pole': ['pole', 'traffic light', 'traffic sign'],
    'background': ['terrain', 'sky'],
    'vehicle': ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
}

cityscapes2concerned_classes = {
    'road': 'road',
    'sidewalk': 'sidewalk',
    'building': 'building',
    'wall': 'building',
    'fence': 'building',
    'pole': 'building',
    'traffic light': 'building',
    'traffic sign': 'building',
    'vegetation': 'vegetation',
    'terrain': 'sidewalk',
    'sky': 'sky',
    'person': 'vehicle',
    'rider': 'vehicle',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'train': 'vehicle',
    'motorcycle': 'vehicle',
    'bicycle': 'vehicle',
}


concerned_classes_list = ['road', 'sidewalk', 'building', 'vegetation', 'sky', 'vehicle']

concerned_classes_ind_map = {cn: i for i, cn in enumerate(concerned_classes_list)}

cityscapes2concerned_index = [
    concerned_classes_ind_map[cityscapes2concerned_classes[cityscapes_classes[i]]] for i in range(len(cityscapes_classes))
]

cityscapes2concerned_index_np = np.array(cityscapes2concerned_index)
cityscapes2concerned_index_torch = torch.tensor(cityscapes2concerned_index)

def get_concerned_classes(cityscapes_class_idx):
    for key, value in concerned_classes_ind_map:
        if cityscapes_classes[cityscapes_class_idx] in concerned_classes[value]:
            return key
    print("Error concerned class.")

def cityscapes2concerned(cityscapes_index_array):
    ret = None
    if isinstance(cityscapes_index_array, np.ndarray):
        ret = cityscapes2concerned_index_np[cityscapes_index_array]
    else:
        assert isinstance(cityscapes_index_array, torch.Tensor)
        ret = cityscapes2concerned_index_torch[cityscapes_index_array]

    assert ret is not None
    return ret

def semantic_prob_to_rgb(semantic_tensor : torch.Tensor):
    assert isinstance(semantic_tensor, torch.Tensor)
    assert semantic_tensor.ndim == 3
    assert semantic_tensor.shape[0] == 6
    pixel_semantic_tag = torch.argmax(semantic_tensor, dim=0)
    rgb = semantic_color_torch.to(semantic_tensor.device)[pixel_semantic_tag].permute(2, 0, 1)

    return rgb

def semantic_tag_to_rgb(pixel_semantic_tag : torch.Tensor):
    assert isinstance(pixel_semantic_tag, torch.Tensor)
    assert pixel_semantic_tag.ndim == 2
    rgb = semantic_color_torch.to(pixel_semantic_tag.device)[pixel_semantic_tag].permute(2, 0, 1) / 255.

    return rgb
