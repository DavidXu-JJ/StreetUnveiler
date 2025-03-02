
#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.semantic_utils import concerned_classes_list

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 K, image, gt_alpha_mask,
                 semantic_map, # [H, W]
                 resize_scale,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        if self.K is not None and resize_scale is not None:
            self.K[:2, :] /= resize_scale
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if semantic_map is not None:
            self.semantic_map = semantic_map.to(self.data_device)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # in submodules/diff-gaussian-rasterization/cuda_rasterization/auxiliary.h
        # transformPoint4x3 and transformPoint4x4 is executed with a transposed matrix
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.c2w = torch.inverse(torch.tensor(getWorld2View2(R, T, trans, scale))).cuda()

        # prospective2orthogonal: project to a cube with z from [0, zfar] (not [-1, 1])
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, K=self.K, img_h=self.image_height, img_w=self.image_width).transpose(0,1).cuda()

        # full_proj_transform need to be (prospective2orthogonal * w2c).T
        # (prospective2orthogonal * w2c).T = (w2c.T * prospective2orthogonal.T)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_certain_semantic_mask(self, semantic_mask_bit):
        semantic_mask = ( (1 << self.semantic_map.to(torch.int32)) & semantic_mask_bit ) > 0
        return semantic_mask

    def get_semantic_prob_image(self):
        semantic_prob_image = torch.zeros((len(concerned_classes_list), self.image_height, self.image_width), dtype=torch.float32).to(self.data_device)
        for i in range(len(concerned_classes_list)):
            semantic_mask = (self.semantic_map == i)
            semantic_prob_image[i, semantic_mask] = 1.0

        return semantic_prob_image


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

