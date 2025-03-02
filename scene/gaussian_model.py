
#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
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
from utils.pcd_utils import SemanticPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.semantic_utils import semantic_color
from utils.disjoint_set_utils import DisjointSet

class GaussianModel:

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


    def __init__(self, sh_degree : int):
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
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._semantics,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._semantics,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def points_number(self):
        return self._xyz.shape[0]

    # initialized as sqrt(mean dist of 3knn)
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    # the norm of quaternion is 1
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    # [point, sphere harmonics, RGB]
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_semantics(self):
        return self._semantics.squeeze(-1)

    @property
    def get_semantics_32bit(self):
        semantics_32bit = (1 << self._semantics.to(torch.int32)).squeeze(-1)
        return semantics_32bit

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : SemanticPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = pcd.points.float().cuda()
        fused_color = RGB2SH(pcd.colors.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(dist3knn(pcd.points.float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # initialized as 0
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantics = pcd.semantics[..., None].to(torch.int32)  # [n, 1]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def requires_grad(self, flag=True):
        self._xyz.requires_grad = flag
        self._features_dc.requires_grad = flag
        self._features_rest.requires_grad = flag
        self._opacity.requires_grad = flag
        self._scaling.requires_grad = flag
        self._rotation.requires_grad = flag

    def eliminate_grad_with_mask(self, mask):
        self._xyz.grad[mask] = 0.
        if self._features_dc.grad is not None:
            self._features_dc.grad[mask] = 0.
        if self._features_rest.grad is not None:
            self._features_rest.grad[mask] = 0.
        self._opacity.grad[mask] = 0.
        self._scaling.grad[mask] = 0.
        self._rotation.grad[mask] = 0.

    def make_splatting_zero_grad(self, mask):
        def modify_zero_grad(x, index_tensor):
            x[index_tensor] = 0
            return x

        selected_zero_grad_indices = torch.arange(self.get_xyz.shape[0])[mask]
        self._xyz.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))
        self._features_dc.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))
        self._features_rest.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))
        self._opacity.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))
        self._scaling.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))
        self._rotation.register_hook(lambda x: modify_zero_grad(x, selected_zero_grad_indices))


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
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

        xyz = self._xyz.detach().cpu().numpy()
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

    def save_rgb_ply(self, path):
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self._features_dc.detach().cpu().squeeze(1).numpy()
        rgb = SH2RGB(features_dc)

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

        xyz = self._xyz.detach().cpu().numpy()
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

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_with_mask(self, mask):
        with torch.no_grad():
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            new_opacities_new = self._opacity
            new_opacities_new[mask] = opacities_new[mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        semantics = np.asarray(plydata.elements[0]['semantics'])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantics = torch.tensor(semantics, dtype=torch.int32, device='cuda')

        self.active_sh_degree = self.max_sh_degree

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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
        else:
            self._xyz = self._xyz[valid_points_mask]
            self._features_dc = self._features_dc[valid_points_mask]
            self._features_rest = self._features_rest[valid_points_mask]
            self._opacity = self._opacity[valid_points_mask]
            self._scaling = self._scaling[valid_points_mask]
            self._rotation = self._rotation[valid_points_mask]
        self._semantics = self._semantics[valid_points_mask]


        if hasattr(self, 'cluster_idx'):
            self.cluster_idx = self.cluster_idx[valid_points_mask]

        if self.xyz_gradient_accum.shape[0] != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        if self.denom.shape[0] != 0:
            self.denom = self.denom[valid_points_mask]

        if self.max_radii2D.shape[0] != 0:
            self.max_radii2D = self.max_radii2D[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantics = torch.cat([self._semantics, new_semantics], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        # means =torch.zeros((stds.size(0), 3),device="cuda")
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantics = self._semantics[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantics)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantics = self._semantics[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics)

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
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def prune_semantic_splatting(self, semantic_mask_bit):
        prune_mask = (self.get_semantics_32bit & semantic_mask_bit) > 0

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def get_semantic_splatting_mask(self, semantic_mask_bit):
        get_mask = (self.get_semantics_32bit & semantic_mask_bit) > 0
        return get_mask

    def get_semantic_index_splatting_mask(self, semantic_index):
        get_mask = self.get_semantics == semantic_index
        return get_mask

    def prune_splatting_with_mask(self, prune_mask):
        assert prune_mask.shape[0] == self.points_number

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def cluster_semantic_instance(self, semantic_mask_bit, threshold = 3e-2, parallel = True):
        dset = DisjointSet(self.points_number).cuda()

        self.cluster_idx = torch.full((self.points_number,), fill_value=-1, dtype=torch.int64).cuda()

        active_points_mask = (self.get_semantics_32bit & semantic_mask_bit) > 0

        indices = torch.arange(self.points_number).cuda()
        active_points_indices = indices[active_points_mask]

        active_points_xyz = self._xyz.detach()[active_points_mask]

        if parallel:
            # This is a parallelized pseudo clustering, but not strictly equivalent to non-parallelized version.
            for i in tqdm(range(active_points_indices.shape[0] - 1)):
                distance_l2 = (torch.abs(active_points_xyz[i] - active_points_xyz) ** 2).sum(dim=-1) ** 0.5
                if_connect = distance_l2 < threshold

                connected_indices = active_points_indices[if_connect]
                grandfather = dset.father[connected_indices].min()
                dset.father[connected_indices] = grandfather
                dset.densify()
            dset.densify()
        else:
            print("Warning: using the slow non-parallelized clustering.")
            for i in tqdm(range(active_points_indices.shape[0]-1)):
                distance_l2 = (torch.abs(active_points_xyz[i]-active_points_xyz[i+1:])**2).sum(dim=-1)**0.5
                if_connect = distance_l2 < threshold

                connected_indices = active_points_indices[i+1:][if_connect]
                for j in connected_indices:
                    dset.union(active_points_indices[i], j)
            dset.densify()

        self.cluster_idx[active_points_mask] = dset.get_father_with_mask(active_points_mask)

        self.prune_invalid_cluster()

    def cluster_instance_with_mask(self, valid_mask, threshold = 7e-2, parallel = True):
        with torch.no_grad():
            dset = DisjointSet(self.points_number).cuda()

            self.cluster_idx = torch.full((self.points_number,), fill_value=-1, dtype=torch.int64).cuda()

            indices = torch.arange(self.points_number).cuda()
            active_points_indices = indices[valid_mask]

            active_points_xyz = self._xyz.detach()[valid_mask]
            if parallel:
                # This is a parallelized pseudo clustering, but not strictly equivalent to non-parallelized version.
                for i in tqdm(range(active_points_indices.shape[0] - 1)):
                    distance_l2 = (torch.abs(active_points_xyz[i] - active_points_xyz) ** 2).sum(dim=-1) ** 0.5
                    if_connect = distance_l2 < threshold

                    connected_indices = active_points_indices[if_connect]
                    grandfather = dset.father[connected_indices].min()
                    dset.father[connected_indices] = grandfather
                    dset.densify()
                dset.densify()
            else:
                print("Warning: using the slow non-parallelized clustering.")
                for i in tqdm(range(active_points_indices.shape[0]-1)):
                    distance_l2 = (torch.abs(active_points_xyz[i]-active_points_xyz[i+1:])**2).sum(dim=-1)**0.5
                    if_connect = distance_l2 < threshold

                    connected_indices = active_points_indices[i+1:][if_connect]
                    for j in connected_indices:
                        dset.union(active_points_indices[i], j)
                dset.densify()

            self.cluster_idx[valid_mask] = dset.get_father_with_mask(valid_mask)

            self.prune_invalid_cluster()

    def get_valid_cluster_mask(self):
        return self.cluster_idx != -1

    """ prune the points without cluster index """
    def prune_invalid_cluster(self):
        self.prune_splatting_with_mask(~self.get_valid_cluster_mask())

    def save_cluster_ply(self, path):
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        cluster_idx_mod = self.cluster_idx.cpu().squeeze(-1).numpy() % len(semantic_color)
        rgb = semantic_color[cluster_idx_mod]

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)

    def save_cluster_idx(self, path, semantic_mask_bit):
        assert hasattr(self, 'cluster_idx')
        torch.save((self.cluster_idx, semantic_mask_bit), path)