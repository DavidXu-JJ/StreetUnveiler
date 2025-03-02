
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
import torch
import torchvision
import os
from tqdm import tqdm
from os import makedirs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gaussian_renderer import render
from scene import Scene
from utils.system_utils import searchForMaxInpaintRound
from scene.env_map import SkyModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_set(model_path, views, gaussians, sky_model, pipeline, background, current_inpaint_round):
    render_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "final_renders")
    depth_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "final_depths")
    rend_normal_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "final_rend_normal")
    surf_normal_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "final_surf_normal")
    gts_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(rend_normal_path, exist_ok=True)
    makedirs(surf_normal_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        sky_image = sky_model.render_with_camera(view.image_height, view.image_width, view.K, view.c2w)
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        render_alpha = render_pkg["rend_alpha"]
        rendering = rendering + sky_image * (1 - render_alpha)
        rend_normal = torch.clamp(render_pkg["rend_normal"] * 0.5 + 0.5, 0.0, 1.0)
        surf_normal = torch.clamp(render_pkg["surf_normal"] * 0.5 + 0.5, 0.0, 1.0)

        depth = torch.clamp(1.0 / render_pkg["surf_depth"], 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rend_normal, os.path.join(rend_normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(surf_normal, os.path.join(surf_normal_path, '{0:05d}'.format(idx) + ".png"))

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, current_inpaint_round):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        sky_model = SkyModel()
        scene = Scene(dataset, gaussians, sky_model, load_iteration=iteration, shuffle=False, only_pose=False,
                      splatting_ply_path=os.path.join(dataset.model_path,
                                                      "instance_workspace_{}".format(current_inpaint_round),
                                                      "checkpoint", "point_cloud.ply")
                      )

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        assert hasattr(scene, 'camera_frame_dict')
        views = scene.getTrainCameras()[scene.camera_frame_dict['front_start']:scene.camera_frame_dict['front_end']]
        render_set(dataset.model_path, views, gaussians, sky_model, pipeline, background, current_inpaint_round)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--current_inpaint_round", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    current_inpaint_round = None
    if args.current_inpaint_round == -1:
        current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path)
    else:
        current_inpaint_round = args.current_inpaint_round

    assert current_inpaint_round is not None

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), current_inpaint_round)