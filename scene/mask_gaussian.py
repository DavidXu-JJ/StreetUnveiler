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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from tqdm import tqdm
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import dist3knn, dist10knn
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.pcd_utils import SemanticPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.semantic_utils import semantic_color
from utils.disjoint_set_utils import DisjointSet

MASK_PROPERTY = ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']
MASK_PROPERTY_BIT = {str: (1 << idx) for idx, str in enumerate(MASK_PROPERTY)}

class MaskGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantics = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.mask = torch.empty(0)
        self._new_xyz = torch.empty(0)
        self._new_features_dc = torch.empty(0)
        self._new_features_rest = torch.empty(0)
        self._new_scaling = torch.empty(0)
        self._new_rotation = torch.empty(0)
        self._new_opacity = torch.empty(0)
        self.setup_functions()

        self.mask_property_bit = 0

    @property
    def check_fixed_xyz(self):
        return MASK_PROPERTY_BIT['xyz'] & self.mask_property_bit != 0

    @property
    def check_fixed_features_dc(self):
        return MASK_PROPERTY_BIT['features_dc'] & self.mask_property_bit != 0

    @property
    def check_fixed_features_rest(self):
        return MASK_PROPERTY_BIT['features_rest'] & self.mask_property_bit != 0

    @property
    def check_fixed_scaling(self):
        return MASK_PROPERTY_BIT['scaling'] & self.mask_property_bit != 0

    @property
    def check_fixed_rotation(self):
        return MASK_PROPERTY_BIT['rotation'] & self.mask_property_bit != 0

    @property
    def check_fixed_opacity(self):
        return MASK_PROPERTY_BIT['opacity'] & self.mask_property_bit != 0

    def set_nograd(self):
        self._xyz.requires_grad=False
        self._features_dc.requires_grad=False
        self._features_rest.requires_grad=False
        self._scaling.requires_grad=False
        self._rotation.requires_grad=False
        self._opacity.requires_grad=False

        self._new_xyz = nn.Parameter(torch.zeros_like(self._xyz).cuda().requires_grad_(True))
        self._new_features_dc = nn.Parameter(torch.zeros_like(self._features_dc).cuda().requires_grad_(True))
        self._new_features_rest = nn.Parameter(torch.zeros_like(self._features_rest).cuda().requires_grad_(True))
        self._new_scaling = nn.Parameter(torch.zeros_like(self._scaling).cuda().requires_grad_(True))
        self._new_rotation = nn.Parameter(torch.zeros_like(self._rotation).cuda().requires_grad_(True))
        self._new_opacity = nn.Parameter(torch.zeros_like(self._opacity).cuda().requires_grad_(True))

        self.set_mask(torch.ones(self.points_number).cuda())

    def set_mask(self, update_mask):
        self.mask = update_mask.cuda()

    def reset_mask(self, update_mask, opt):
        self._xyz = self.get_xyz.detach()
        self._features_dc = (self._features_dc + self._new_features_dc * self.mask[..., None, None]).detach()
        self._features_rest = (self._features_rest + self._new_features_rest * self.mask[..., None, None]).detach()
        self._scaling = (self._scaling + self._new_scaling * self.mask[..., None]).detach()
        self._rotation = (self._rotation + self._new_rotation * self.mask[..., None]).detach()
        self._opacity = (self._opacity + self._new_opacity * self.mask[..., None]).detach()
        self.mask = update_mask

        self.set_nograd()
        self.training_setup(opt)

    @property
    def points_number(self):
        return self._xyz.shape[0]

    # initialized as sqrt(mean dist of 3knn)
    @property
    def get_scaling(self):
        if self.check_fixed_scaling:
             return self.scaling_activation(self._scaling)
        return self.scaling_activation(self._scaling + self._new_scaling * self.mask[..., None])

    # the norm of quaternion is 1
    @property
    def get_rotation(self):
        if self.check_fixed_rotation:
            return self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation + self._new_rotation * self.mask[..., None])

    @property
    def get_xyz(self):
        if self.check_fixed_xyz:
            return self._xyz
        return self._xyz + self._new_xyz * self.mask[..., None]

    # [point, sphere harmonics, RGB]
    @property
    def get_features(self):
        if self.check_fixed_features_dc:
            features_dc = self._features_dc
        else:
            features_dc = self._features_dc + self._new_features_dc * self.mask[..., None, None]

        if self.check_fixed_features_rest:
            features_rest = self._features_rest
        else:
            features_rest = self._features_rest + self._new_features_rest * self.mask[..., None, None]
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self.check_fixed_opacity:
            return self.opacity_activation(self._opacity)
        return self.opacity_activation(self._opacity + self._new_opacity * self.mask[..., None])

    @property
    def get_semantics(self):
        return self._semantics.squeeze(-1)

    @property
    def get_semantics_32bit(self):
        semantics_32bit = (1 << self._semantics.to(torch.int32)).squeeze(-1)
        return semantics_32bit

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier,
                                          self._rotation + self._new_rotation * self.mask[..., None])

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._new_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "new_xyz"},
            {'params': [self._new_features_dc], 'lr': training_args.feature_lr, "name": "new_f_dc"},
            {'params': [self._new_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "new_f_rest"},
            {'params': [self._new_opacity], 'lr': training_args.opacity_lr, "name": "new_opacity"},
            {'params': [self._new_scaling], 'lr': training_args.scaling_lr, "name": "new_scaling"},
            {'params': [self._new_rotation], 'lr': training_args.rotation_lr, "name": "new_rotation"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "new_xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def from_gaussian_model(self, gaussians: GaussianModel):
        self._xyz = gaussians._xyz.detach().clone().cuda()
        self._features_dc = gaussians._features_dc.detach().clone().cuda()
        self._features_rest = gaussians._features_rest.detach().clone().cuda()
        self._opacity = gaussians._opacity.detach().clone().cuda()
        self._scaling = gaussians._scaling.detach().clone().cuda()
        self._rotation = gaussians._rotation.detach().clone().cuda()
        self._semantics = gaussians._semantics.cuda()
        self.max_radii2D = gaussians.max_radii2D.cuda()
        self.active_sh_degree = gaussians.active_sh_degree
        self.spatial_lr_scale = gaussians.spatial_lr_scale
        self.set_nograd()

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = (self._xyz + self._new_xyz * self.mask[..., None]).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (self._features_dc + self._new_features_dc * self.mask[..., None, None]).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = (self._features_rest + self._new_features_rest * self.mask[..., None, None]).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = (self._opacity + self._new_opacity * self.mask[..., None]).detach().cpu().numpy()
        scale = (self._scaling + self._new_scaling * self.mask[..., None]).detach().cpu().numpy()
        rotation = (self._rotation + self._new_rotation * self.mask[..., None]).detach().cpu().numpy()
        semantics = self._semantics.detach().cpu().numpy().astype(np.int32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dtype_full += [('semantics', 'i4')]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantics), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_semantic_ply(self, path):
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        mkdir_p(os.path.dirname(path))

        xyz = (self._xyz + self._new_xyz * self.mask[..., None]).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        semantics = self._semantics.cpu().squeeze(-1).numpy()
        rgb = semantic_color[semantics]

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)

    def save_opacity_ply(self, path):
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        mkdir_p(os.path.dirname(path))

        xyz = (self._xyz + self._new_xyz * self.mask[..., None]).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacity = self.get_opacity.repeat(1, 3).detach().cpu().squeeze(-1).numpy()
        rgb = opacity * 255.
        rgb[..., 1:3] = 0.

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, min_opacity):
        prune_mask = self.mask * (self.get_opacity < min_opacity).squeeze()
        valid_points_mask = ~prune_mask
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._new_xyz = optimizable_tensors["new_xyz"]
            self._new_features_dc = optimizable_tensors["new_f_dc"]
            self._new_features_rest = optimizable_tensors["new_f_rest"]
            self._new_opacity = optimizable_tensors["new_opacity"]
            self._new_scaling = optimizable_tensors["new_scaling"]
            self._new_rotation = optimizable_tensors["new_rotation"]
        else:
            self._new_xyz = self._new_xyz[valid_points_mask]
            self._new_features_dc = self._new_features_dc[valid_points_mask]
            self._new_features_rest = self._new_features_rest[valid_points_mask]
            self._new_opacity = self._new_opacity[valid_points_mask]
            self._new_scaling = self._new_scaling[valid_points_mask]
            self._new_rotation = self._new_rotation[valid_points_mask]

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        self._semantics = self._semantics[valid_points_mask]
        self.mask = self.mask[valid_points_mask]

        if self.xyz_gradient_accum.shape[0] != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        if self.denom.shape[0] != 0:
            self.denom = self.denom[valid_points_mask]

        if self.max_radii2D.shape[0] != 0:
            self.max_radii2D = self.max_radii2D[valid_points_mask]

        if hasattr(self, "next_editable_pcd_mask"):
            self.next_editable_pcd_mask = self.next_editable_pcd_mask[valid_points_mask]

        if hasattr(self, "already_in_frame_mask"):
            self.already_in_frame_mask = self.already_in_frame_mask[valid_points_mask]

        return prune_mask

    def prune_points_with_mask(self, prune_mask):
        valid_points_mask = ~prune_mask
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._new_xyz = optimizable_tensors["new_xyz"]
            self._new_features_dc = optimizable_tensors["new_f_dc"]
            self._new_features_rest = optimizable_tensors["new_f_rest"]
            self._new_opacity = optimizable_tensors["new_opacity"]
            self._new_scaling = optimizable_tensors["new_scaling"]
            self._new_rotation = optimizable_tensors["new_rotation"]
        else:
            self._new_xyz = self._new_xyz[valid_points_mask]
            self._new_features_dc = self._new_features_dc[valid_points_mask]
            self._new_features_rest = self._new_features_rest[valid_points_mask]
            self._new_opacity = self._new_opacity[valid_points_mask]
            self._new_scaling = self._new_scaling[valid_points_mask]
            self._new_rotation = self._new_rotation[valid_points_mask]

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        self._semantics = self._semantics[valid_points_mask]
        self.mask = self.mask[valid_points_mask]

        if self.xyz_gradient_accum.shape[0] != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        if self.denom.shape[0] != 0:
            self.denom = self.denom[valid_points_mask]

        if self.max_radii2D.shape[0] != 0:
            self.max_radii2D = self.max_radii2D[valid_points_mask]

        if hasattr(self, "next_editable_pcd_mask"):
            self.next_editable_pcd_mask = self.next_editable_pcd_mask[valid_points_mask]

        if hasattr(self, "already_in_frame_mask"):
            self.already_in_frame_mask = self.already_in_frame_mask[valid_points_mask]

        return prune_mask

    def reset_opacity(self):
        with torch.no_grad():
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            new_opacities_new = self._new_opacity
            new_opacities_new[self.mask] = (opacities_new - self._opacity)[self.mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacities_new, "new_opacity")
        self._new_opacity = optimizable_tensors["new_opacity"]

    def reset_opacity_with_mask(self, mask):
        with torch.no_grad():
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            new_opacities_new = self._new_opacity
            new_opacities_new[mask] = (opacities_new - self._opacity)[mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacities_new, "new_opacity")
        self._new_opacity = optimizable_tensors["new_opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                    if "exp_avg_sq" in stored_state:
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]

                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
            self,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_semantics,
            new_xyz_new,
            new_features_dc_new,
            new_features_rest_new,
            new_opacities_new,
            new_scaling_new,
            new_rotation_new,
            new_mask,
    ):
        d = {"new_xyz": new_xyz_new,
        "new_f_dc": new_features_dc_new,
        "new_f_rest": new_features_rest_new,
        "new_opacity": new_opacities_new,
        "new_scaling" : new_scaling_new,
        "new_rotation" : new_rotation_new}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._new_xyz = optimizable_tensors["new_xyz"]
        self._new_features_dc = optimizable_tensors["new_f_dc"]
        self._new_features_rest = optimizable_tensors["new_f_rest"]
        self._new_opacity = optimizable_tensors["new_opacity"]
        self._new_scaling = optimizable_tensors["new_scaling"]
        self._new_rotation = optimizable_tensors["new_rotation"]
        self._xyz = torch.cat([self._xyz, new_xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_features_rest], dim=0)
        self._opacity = torch.cat([self._opacity, new_opacities], dim=0)
        self._scaling = torch.cat([self._scaling, new_scaling], dim=0)
        self._rotation = torch.cat([self._rotation, new_rotation], dim=0)
        self._semantics = torch.cat([self._semantics, new_semantics], dim=0)

        self.mask = torch.cat([self.mask, new_mask], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if hasattr(self, "next_editable_pcd_mask"):
            self.next_editable_pcd_mask = torch.cat([self.next_editable_pcd_mask, torch.ones(new_xyz.shape[0], device="cuda").bool()], dim=0)

        if hasattr(self, "already_in_frame_mask"):
            self.already_in_frame_mask = torch.cat([self.already_in_frame_mask, torch.ones(new_xyz.shape[0], device="cuda").bool()], dim=0)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        selected_pts_mask = selected_pts_mask * self.mask

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantics = self._semantics[selected_pts_mask]

        new_xyz_new = self._new_xyz[selected_pts_mask]
        new_features_dc_new = self._new_features_dc[selected_pts_mask]
        new_features_rest_new = self._new_features_rest[selected_pts_mask]
        new_opacities_new = self._new_opacity[selected_pts_mask]
        new_scaling_new = self._new_scaling[selected_pts_mask]
        new_rotation_new = self._new_rotation[selected_pts_mask]

        new_mask = self.mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_semantics, new_xyz_new, new_features_dc_new, new_features_rest_new,
                                   new_opacities_new, new_scaling_new, new_rotation_new, new_mask)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        selected_pts_mask = selected_pts_mask * self.mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask] + self._new_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantics = self._semantics[selected_pts_mask].repeat(N,1)

        new_xyz_new = torch.zeros_like(new_xyz)
        new_scaling_new = self._new_scaling[selected_pts_mask].repeat(N,1)
        new_rotation_new = self._new_rotation[selected_pts_mask].repeat(N,1)
        new_features_dc_new = self._new_features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest_new = self._new_features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacities_new = self._new_opacity[selected_pts_mask].repeat(N,1)

        new_mask = self.mask[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantics,
                                   new_xyz_new, new_features_dc_new, new_features_rest_new, new_opacities_new, new_scaling_new, new_rotation_new, new_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points_with_mask(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points_with_mask(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        update_filter = self.mask * update_filter
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def set_fixed_xyz(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['xyz']

    def set_fixed_features_dc(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['features_dc']

    def set_fixed_features_rest(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['features_rest']

    def set_fixed_scaling(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['scaling']

    def set_fixed_rotation(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['rotation']

    def set_fixed_opacity(self):
        self.mask_property_bit = self.mask_property_bit | MASK_PROPERTY_BIT['opacity']
