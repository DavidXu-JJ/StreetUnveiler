
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

import sys
import os
import numpy as np
import torch
import torchvision
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
from enum import Enum
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.system_utils import searchForMaxIteration, searchForMaxInpaintRound
from scene.env_map import SkyModel
from gaussian_renderer import render, render_with_mask
from simple_knn._C import meanDistFromReferencePcd
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from inpainting_pipeline.utils import dilate_mask

class InpaintRGBType(Enum):
    GT = "gt"
    RENDER = "render"
    RENDER_WO_INSTANCE = "render_wo_instance"

inpainted_rgb_type = InpaintRGBType.RENDER_WO_INSTANCE

def include_neighbor_pcd(
        dataset: ModelParams,
        load_iteration: int,
        removed_pcd_mask,
        current_inpaint_round: int,
):
    gaussians = GaussianModel(dataset.sh_degree)
    if current_inpaint_round > 0:
        last_inpaint_checkpoint = os.path.join(
            dataset.model_path,
            "instance_workspace_{}".format(current_inpaint_round - 1),
            "checkpoint"
        )

        last_inpaint_pcd = os.path.join(last_inpaint_checkpoint, "point_cloud.ply")
        gaussians.load_ply(last_inpaint_pcd)

    else:
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "checkpoint"))
        else:
            loaded_iter = load_iteration

        gaussians.load_ply(
            os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")
        )

    instance_workspace = os.path.join(dataset.model_path, "instance_workspace_{}".format(current_inpaint_round))

    removed_pcd = gaussians.get_xyz[removed_pcd_mask]

    other_pcd = gaussians.get_xyz[~removed_pcd_mask]
    means_distance = meanDistFromReferencePcd(
        other_pcd, removed_pcd, False
    )

    # Set the trainable mask of the gauusian model.
    # This mask is used for the initialization of scene.mask_gaussian.
    far_mask = means_distance > 4e-2
    temp = (~removed_pcd_mask).clone()
    temp[temp == True] = far_mask
    pcd_mask = ~(temp.clone())
    torch.save(pcd_mask, os.path.join(instance_workspace, "trainable_pcd_mask.pt"))

    # far_mask = means_distance > 5e-3
    far_mask = means_distance > 2e-2
    temp = (~removed_pcd_mask).clone()
    temp[temp == True] = far_mask
    pcd_mask = ~(temp.clone())

    torch.save(pcd_mask, os.path.join(instance_workspace, "editable_pcd_mask.pt"))

    return pcd_mask


def render_set(model_path, views, gaussians, pipeline, background, removed_pcd_mask, current_inpaint_workspace, sky_model):
    mask_path = os.path.join(current_inpaint_workspace, "mask_inpaint")
    inpainted_depth_path = os.path.join(current_inpaint_workspace, "inpainted_depth")
    inpainted_rgb_path = os.path.join(current_inpaint_workspace, "inpainted_rgb")
    inpainted_normal_path = os.path.join(current_inpaint_workspace, "inpainted_normal")
    original_rgb_path = os.path.join(current_inpaint_workspace, "original_rgb")
    empty_opacity_path = os.path.join(current_inpaint_workspace, "empty_opacity")

    makedirs(mask_path, exist_ok=True)
    makedirs(inpainted_depth_path, exist_ok=True)
    makedirs(inpainted_rgb_path, exist_ok=True)
    makedirs(inpainted_normal_path, exist_ok=True)
    makedirs(original_rgb_path, exist_ok=True)
    makedirs(empty_opacity_path, exist_ok=True)

    valid_frame_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        sky_image = sky_model.render_with_camera(view.image_height, view.image_width, view.K, view.c2w)
        render_pkg = render(view, gaussians, pipeline, background)
        rendered_alpha = render_pkg["rend_alpha"]
        original_rgb = render_pkg['render'] + sky_image * (1 - rendered_alpha)
        torchvision.utils.save_image(original_rgb, os.path.join(original_rgb_path, '{0:05d}'.format(idx) + ".png"))

        bg_render_pkg = render_with_mask(view, gaussians, pipeline, background, ~removed_pcd_mask)
        background_rgb = bg_render_pkg['render'] + sky_image * (1 - rendered_alpha)

        removed_rendered_alpha = bg_render_pkg["rend_alpha"]

        torchvision.utils.save_image(rendered_alpha - removed_rendered_alpha, os.path.join(empty_opacity_path, '{0:05d}'.format(idx) + ".png"))

        depth = None
        if inpainted_rgb_type == InpaintRGBType.GT or inpainted_rgb_type == InpaintRGBType.RENDER:
            depth = render_pkg['surf_depth']
        elif inpainted_rgb_type == InpaintRGBType.RENDER_WO_INSTANCE:
            depth = bg_render_pkg['surf_depth']
        assert depth is not None

        threshold = 0.01
        instance_pix_mask = (torch.abs(rendered_alpha - removed_rendered_alpha) > threshold).sum(0)

        valid_frame_list.append(idx)
        instance_pix_mask = dilate_mask(instance_pix_mask, 5).cpu()
        torchvision.utils.save_image(instance_pix_mask.float(), os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        np.save(os.path.join(mask_path, '{0:05d}'.format(idx) + ".npy"), instance_pix_mask.numpy())

        raw_disparity = (1.0 / depth)
        raw_disparity[raw_disparity.isinf()] = 0.0
        disparity = torch.clamp(raw_disparity, 0.0, 1.0)
        torchvision.utils.save_image(disparity, os.path.join(inpainted_depth_path, '{0:05d}'.format(idx) + ".png"))

        if inpainted_rgb_type == InpaintRGBType.GT:
            torchvision.utils.save_image(view.original_image, os.path.join(inpainted_rgb_path, '{0:05d}'.format(idx) + ".png"))
        elif inpainted_rgb_type == InpaintRGBType.RENDER:
            torchvision.utils.save_image(original_rgb, os.path.join(inpainted_rgb_path, '{0:05d}'.format(idx) + ".png"))
        elif inpainted_rgb_type == InpaintRGBType.RENDER_WO_INSTANCE:
            torchvision.utils.save_image(background_rgb, os.path.join(inpainted_rgb_path, '{0:05d}'.format(idx) + ".png"))

        normal = None
        if inpainted_rgb_type == InpaintRGBType.GT or inpainted_rgb_type == InpaintRGBType.RENDER:
            normal = render_pkg['rend_normal']
        elif inpainted_rgb_type == InpaintRGBType.RENDER_WO_INSTANCE:
            normal = bg_render_pkg['rend_normal']
        assert normal is not None

        normal = torch.clamp(normal * 0.5 + 0.5, 0.0, 1.0)
        torchvision.utils.save_image(normal, os.path.join(inpainted_normal_path, '{0:05d}'.format(idx) + ".png"))

    valid_frame_list = np.array(valid_frame_list)
    np.save(os.path.join(current_inpaint_workspace, 'valid_inpaint_frame.npy'), valid_frame_list)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, pcd_mask, current_inpaint_workspace):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        sky_model = SkyModel()
        scene = Scene(dataset, gaussians, sky_model, load_iteration=iteration, shuffle=False, only_pose=True)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        assert hasattr(scene, 'camera_frame_dict')
        views = scene.getTrainCameras()[scene.camera_frame_dict['front_start']:scene.camera_frame_dict['front_end']]
        render_set(dataset.model_path, views, gaussians, pipeline, background, pcd_mask, current_inpaint_workspace, sky_model)


if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--mask_path", default="", type=str)
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--current_inpaint_round", default=-1, type=int)
    args = get_combined_args(parser)

    current_inpaint_round = None
    if args.current_inpaint_round == -1:
        current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path)
    else:
        current_inpaint_round = args.current_inpaint_round

    assert current_inpaint_round is not None

    current_inpaint_workspace = os.path.join(model.extract(args).model_path, "instance_workspace_{}".format(current_inpaint_round))

    if args.mask_path:
        mask_path = args.mask_path
    else:
        mask_path = os.path.join(current_inpaint_workspace, "removed_pcd_mask.pt")

    removed_pcd_mask = torch.load(mask_path).cuda()

    removed_pcd_mask = include_neighbor_pcd(
        model.extract(args),
        args.load_iteration,
        removed_pcd_mask,
        current_inpaint_round,
    )

    render_sets(model.extract(args), args.load_iteration, pipeline.extract(args), removed_pcd_mask, current_inpaint_workspace)
