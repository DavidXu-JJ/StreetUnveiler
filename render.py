#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_semantic
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.system_utils import searchForMaxInpaintRound
from scene.env_map import SkyModel

import open3d as o3d

def render_set(model_path, name, scene, gaussians, pipeline, background, sky_model):
    iteration = scene.loaded_iter
    views = scene.getTrainCameras()
    if name == "test":
        views = scene.getTestCameras()
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    rendered_semantics_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_semantics")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_render")
    normal_surf_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_surf")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(rendered_semantics_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_render_path, exist_ok=True)
    makedirs(normal_surf_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        select_frame_id = idx
        sky_image = sky_model.render_with_camera(view.image_height, view.image_width, view.K, view.c2w)
        render_pkg = render(view, gaussians, pipeline, background)
        semantic_pkg = render_semantic(view, gaussians, pipeline, background)
        semantic_rgb = semantic_pkg['semantic_rgb']
        torchvision.utils.save_image(semantic_rgb, os.path.join(rendered_semantics_path, '{0:05d}'.format(idx) + ".png"))

        rendering = render_pkg["render"] + sky_image * (1 - render_pkg["rend_alpha"])
        depth = torch.clamp((1.0 / render_pkg["surf_depth"]).nan_to_num(), 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
        torchvision.utils.save_image(rend_normal, os.path.join(normal_render_path, '{0:05d}'.format(idx) + ".png"))
        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
        torchvision.utils.save_image(surf_normal, os.path.join(normal_surf_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        sky_model = SkyModel()
        scene = Scene(dataset, gaussians, sky_model, load_iteration=iteration, shuffle=False, only_pose=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene, gaussians, pipeline, background, sky_model)

        if not skip_test:
             render_set(dataset.model_path, "test", scene, gaussians, pipeline, background, sky_model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=0.004, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=3.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--num_cluster", default=1000, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    sky_model = SkyModel()
    if current_inpaint_round != -1:
        scene = Scene(dataset, gaussians, sky_model, load_iteration=iteration, shuffle=False, only_pose=True,
                      splatting_ply_path=os.path.join(dataset.model_path,
                                                      "instance_workspace_{}".format(current_inpaint_round),
                                                      "checkpoint", "point_cloud.ply"))
    else:
        scene = Scene(dataset, gaussians, sky_model, load_iteration=iteration, shuffle=False, only_pose=True,
                        splatting_ply_path=os.path.join(dataset.model_path,
                                                        "point_cloud",
                                                        "iteration_50000", "point_cloud.ply"))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    # train_dir = os.path.join(args.model_path, 'original_mesh', "ours_{}".format(scene.loaded_iter))
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        camera_length = len(scene.getTrainCameras())
        # gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.reconstruction(scene.getTrainCameras()[0:camera_length//3])
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=args.voxel_size, sdf_trunc=5 * args.voxel_size,
                                                       depth_trunc=args.depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))