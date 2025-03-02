
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

from collections import defaultdict
import numpy as np
import torch

from utils.general_utils import to_torch

class SemanticPointCloud():
    def __init__(
            self,
            points,
            colors=None,
            normals=None,
            semantics=None,
            semantics_dict=None,
    ):
        # [n, 3]
        self.points = to_torch(points)
        # [n, 3]
        self.colors = to_torch(colors)
        # [n, 3]
        self.normals = to_torch(normals)
        # [n]
        self.semantics = to_torch(semantics)
        if self.semantics is not None:
            self.semantics = self.semantics.to(torch.int32).reshape(-1)

        # dict[str] = int
        self.semantics_dict = semantics_dict

        assert (
                (self.colors is None or self.points.shape[0] == self.colors.shape[0]) and
                (self.normals is None or self.points.shape[0] == self.normals.shape[0]) and
                (self.semantics is None or self.points.shape[0] == self.semantics.shape[0])
        )

    def load_semantics(self, semantics):
        temp = to_torch(semantics).to(torch.int32)
        assert temp.shape[0] == self.points.shape[0]
        self.semantics = temp

    def compute_min_bound(self):
        if self.points.shape[0] == 0:
            return torch.Tensor([0., 0., 0.]).to(self.points.device)

        return torch.min(self.points, dim=0)[0].to(self.points.device)

    def compute_max_bound(self):
        if self.points.shape[0] == 0:
            return torch.Tensor([0., 0., 0.]).to(self.points.device)

        return torch.max(self.points, dim=0)[0].to(self.points.device)

    def compute_voxel_index(self, ref_coords, offset: int):
        int_ref_coords = ref_coords.to(torch.int64)

        offset_matrix = torch.tensor([1, offset, offset * offset], dtype=torch.int64).to(int_ref_coords.device)

        # int32 will lose the precision
        return torch.sum((int_ref_coords * offset_matrix).to(torch.int64), dim=-1).to(torch.int64)

    def voxel_down_sample(self, voxel_size: float):
        print("The semantic point cloud downsampling...")
        min_bound = self.compute_min_bound() - voxel_size * 0.5
        max_bound = self.compute_max_bound() + voxel_size * 0.5

        offset = int((torch.max(max_bound).item() - torch.min(min_bound).item() + 2 * voxel_size) / voxel_size)

        ref_coords = (self.points - min_bound) / voxel_size
        # [n]
        voxel_index = self.compute_voxel_index(ref_coords, offset)

        sorted_values, sorted_indices = torch.sort(voxel_index)

        unique_voxel_index, counts = torch.unique(sorted_values, return_counts=True)

        split_sizes = tuple(counts.tolist())
        split_indices = torch.split(torch.arange(voxel_index.size(0)), split_sizes)

        return_points = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.points.device)
        count = 0
        sorted_points = self.points[sorted_indices]
        for index in split_indices:
            return_points[count] = sorted_points[index].mean(0)
            count += 1
        assert count == unique_voxel_index.shape[0]

        return_colors = None
        if self.colors is not None:
            return_colors = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.colors.device)
            count = 0
            sorted_colors = self.colors[sorted_indices]
            for index in split_indices:
                return_colors[count] = sorted_colors[index].mean(0)
                count += 1
            assert count == unique_voxel_index.shape[0]

        return_normals = None
        if self.normals is not None:
            return_normals = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.normals.device)
            count = 0
            sorted_normals = self.normals[sorted_indices]
            for index in split_indices:
                return_normals[count] = sorted_normals[index].mean(0)
                count += 1
            assert count == unique_voxel_index.shape[0]

        valid_semantic_mask = torch.ones((unique_voxel_index.shape[0]), dtype=torch.bool)
        return_semantic = None
        if self.semantics is not None:
            return_semantic = torch.zeros((unique_voxel_index.shape[0])).to(torch.int32).to(self.semantics.device)
            count = 0
            sorted_semantics = self.semantics[sorted_indices]
            for index in split_indices:
                current_voxel_semantics = sorted_semantics[index]
                return_semantic[count] = torch.mode(current_voxel_semantics, dim=0).values.item()
                unique_test_elements, semantic_counts = torch.unique(current_voxel_semantics, return_counts=True)
                if 1.0 * semantic_counts.max() / len(current_voxel_semantics) < 0.8:
                    valid_semantic_mask[count] = False
                count += 1
            assert count == unique_voxel_index.shape[0]

        return_points = return_points[valid_semantic_mask]
        if return_colors is not None:
            return_colors = return_colors[valid_semantic_mask]
        if return_normals is not None:
            return_normals = return_normals[valid_semantic_mask]
        if return_semantic is not None:
            return_semantic = return_semantic[valid_semantic_mask]
        print("The semantic point cloud downsampling finished.")
        return SemanticPointCloud(points=return_points, colors=return_colors, normals=return_normals, semantics=return_semantic, semantics_dict=self.semantics_dict)

    def voxel_down_sample_with_no_grad(self, voxel_size: float):
        print("The semantic point cloud downsampling...")
        min_bound = self.compute_min_bound() - voxel_size * 0.5
        max_bound = self.compute_max_bound() + voxel_size * 0.5

        offset = int((torch.max(max_bound).item() - torch.min(min_bound).item() + 2 * voxel_size) / voxel_size)

        ref_coords = (self.points - min_bound) / voxel_size
        # [n]
        voxel_index = self.compute_voxel_index(ref_coords, offset)

        sorted_values, sorted_indices = torch.sort(voxel_index)

        unique_voxel_index, counts = torch.unique(sorted_values, return_counts=True)

        split_sizes = tuple(counts.tolist())
        split_indices = torch.split(torch.arange(voxel_index.size(0)), split_sizes)

        return_points = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.points.device)
        count = 0
        sorted_points = self.points[sorted_indices].detach()
        for index in split_indices:
            return_points[count] = sorted_points[index].mean(0)
            count += 1
        assert count == unique_voxel_index.shape[0]

        return_colors = None
        if self.colors is not None:
            return_colors = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.colors.device)
            count = 0
            sorted_colors = self.colors[sorted_indices].detach()
            for index in split_indices:
                return_colors[count] = sorted_colors[index].mean(0)
                count += 1
            assert count == unique_voxel_index.shape[0]

        return_normals = None
        if self.normals is not None:
            return_normals = torch.zeros((unique_voxel_index.shape[0], 3)).to(self.normals.device)
            count = 0
            sorted_normals = self.normals[sorted_indices].detach()
            for index in split_indices:
                return_normals[count] = sorted_normals[index].mean(0)
                count += 1
            assert count == unique_voxel_index.shape[0]

        valid_semantic_mask = torch.ones((unique_voxel_index.shape[0]), dtype=torch.bool)
        return_semantic = None
        if self.semantics is not None:
            return_semantic = torch.zeros((unique_voxel_index.shape[0])).to(torch.int32).to(self.semantics.device)
            count = 0
            sorted_semantics = self.semantics[sorted_indices].detach()
            for index in split_indices:
                current_voxel_semantics = sorted_semantics[index]
                return_semantic[count] = torch.mode(current_voxel_semantics, dim=0).values.item()
                unique_test_elements, semantic_counts = torch.unique(current_voxel_semantics, return_counts=True)
                if 1.0 * semantic_counts.max() / len(current_voxel_semantics) < 0.8:
                    valid_semantic_mask[count] = False
                count += 1
            assert count == unique_voxel_index.shape[0]

        return_points = return_points[valid_semantic_mask]
        if return_colors is not None:
            return_colors = return_colors[valid_semantic_mask]
        if return_normals is not None:
            return_normals = return_normals[valid_semantic_mask]
        if return_semantic is not None:
            return_semantic = return_semantic[valid_semantic_mask]
        print("The semantic point cloud downsampling finished.")
        return SemanticPointCloud(points=return_points, colors=return_colors, normals=return_normals, semantics=return_semantic, semantics_dict=self.semantics_dict)

