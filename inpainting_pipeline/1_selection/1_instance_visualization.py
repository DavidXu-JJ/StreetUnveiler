
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
import torch
import torchvision
from os import makedirs
from tqdm import tqdm
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.semantic_utils import concerned_classes_ind_map
from utils.system_utils import searchForMaxIteration, searchForMaxInpaintRound
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render_with_mask

def semantic_cluster(
        dataset: ModelParams,
        load_iteration: int,
        semantic_mask_bit: int, # This is a bit mask of the classes to be considered.
        reverse_semantic: bool, # Whether to reverse the semantic bit-mask.
        current_inpaint_round: int,
):
    """
        Cluster the instance with the semantic mask.
    """
    gaussians = GaussianModel(dataset.sh_degree)
    if current_inpaint_round > 0:
        last_inpaint_checkpoint = os.path.join(
            dataset.model_path,
            "instance_workspace_{}".format(current_inpaint_round - 1),
            "checkpoint"
        )

        inpainted_pcd = os.path.join(last_inpaint_checkpoint, "point_cloud.ply")
        gaussians.load_ply(inpainted_pcd)
    else:
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "checkpoint"))
        else:
            loaded_iter = load_iteration

        gaussians.load_ply(
            os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")
        )

    if reverse_semantic:
        semantic_mask_bit = ~semantic_mask_bit

    semantic_mask = gaussians.get_semantic_splatting_mask(semantic_mask_bit)

    valid_mask = semantic_mask

    print("start clustering...")

    # Important: After this step, each gaussian point is assigned to a cluster through a cluster_idx on each gaussian.
    gaussians.cluster_instance_with_mask(valid_mask)

    cluster_path = os.path.join(dataset.model_path, "instance_workspace_{}".format(current_inpaint_round))
    makedirs(cluster_path, exist_ok=True)
    gaussians.save_cluster_ply(os.path.join(cluster_path, "cluster.ply"))
    # Save the cluster_idx of each gaussian, and the corresponding semantic info.
    gaussians.save_cluster_idx(os.path.join(cluster_path, "cluster_idx.pt"), semantic_mask_bit)

    return gaussians.cluster_idx.detach().clone(), semantic_mask_bit, cluster_path

def render_set(model_path, views, gaussians, scene, pipeline, background, final_mask, cluster_idx, current_inpaint_round):
    render_path = os.path.join(model_path, "instance_workspace_{}".format(current_inpaint_round), "instance_render")

    makedirs(render_path, exist_ok=True)

    with torch.no_grad():
        current_pcd = gaussians.get_xyz[final_mask]
        pick_id = -1
        for idx in range(len(views)):
            valid_pix, valid_depth = scene.getPcdPixelCoordsInTrainFrameWithDepth(current_pcd, idx)
            valid_nums = valid_depth.shape[0]
            mean_depth = valid_depth.mean()
            if valid_nums > 0.9 * current_pcd.shape[0] and mean_depth < 4.:
                pick_id = idx
                break

        if pick_id == -1:
            for idx in range(len(views)):
                valid_pix, valid_depth = scene.getPcdPixelCoordsInTrainFrameWithDepth(current_pcd, idx)
                valid_nums = valid_depth.shape[0]
                if valid_nums > 0.5 * current_pcd.shape[0]:
                    pick_id = idx
                    break

        render_pkg = render_with_mask(views[pick_id], gaussians, pipeline, background, final_mask)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(cluster_idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, current_inpaint_round : int, cluster_idx, semantic_filter_bit, cluster_path):
    """
        Render each instance in the cluster.
    """
    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        if current_inpaint_round > 0:
            last_inpaint_checkpoint = os.path.join(dataset.model_path, "instance_workspace_{}".format(current_inpaint_round - 1), "checkpoint")
            scene = Scene(
                dataset, gaussians, load_iteration=iteration, shuffle=False, only_pose=True,
                splatting_ply_path= os.path.join(last_inpaint_checkpoint, "point_cloud.ply")
            )
        else:
            scene = Scene(
                dataset, gaussians, load_iteration=iteration, shuffle=False, only_pose=True,
            )

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cluster_idx = cluster_idx.cuda()

        semantic_mask = gaussians.get_semantic_splatting_mask(semantic_filter_bit)

        active_mask = semantic_mask

        unique, counts = torch.unique(cluster_idx, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)

        sorted_unique = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        k = torch.where(sorted_counts > 50)[0][-1]
        topk_elements = sorted_unique[:k]
        topk_counts = sorted_counts[:k]

        solid_cluster_mask = torch.zeros_like(active_mask)

        for idx in tqdm(topk_elements):
            final_mask = (cluster_idx == idx)
            temp = active_mask.clone()
            temp[temp == True] = final_mask
            final_mask = temp
            solid_cluster_mask |= final_mask
            render_set(dataset.model_path, scene.getTrainCameras(), gaussians, scene, pipeline, background, final_mask, idx, current_inpaint_round)

        # Some noisy points can be filtered out through this step.
        torch.save(solid_cluster_mask, os.path.join(cluster_path, "solid_cluster_mask.pt"))
        gaussians.prune_splatting_with_mask(~solid_cluster_mask)
        gaussians.save_rgb_ply(os.path.join(cluster_path, "solid_cluster.ply"))


if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iteration", default=-1, type=int)
    # This pipeline may be run multiple times if necessary, so we need to keep track of the inpaint round.
    parser.add_argument("--current_inpaint_round", default=-1, type=int)
    parser.add_argument("--mask_vehicle", action="store_true")
    parser.add_argument("--mask_vegetation", action="store_true")
    parser.add_argument("--reverse_semantic", action="store_true")
    args = get_combined_args(parser)

    semantic_mask_bit = None
    if args.mask_vehicle or args.mask_vegetation:
        semantic_mask_bit = (
            (1 if args.mask_vehicle else 0) << concerned_classes_ind_map['vehicle'] |
            (1 if args.mask_vegetation else 0) << concerned_classes_ind_map['vegetation']
        )

    assert semantic_mask_bit is not None

    current_inpaint_round = None
    if args.current_inpaint_round == -1:
        current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path) + 1
    else:
        current_inpaint_round = args.current_inpaint_round

    assert current_inpaint_round is not None

    cluster_idx, semantic_mask_bit, cluster_path = semantic_cluster(
        model.extract(args),
        args.load_iteration,
        semantic_mask_bit,
        args.reverse_semantic,
        current_inpaint_round,
    )

    render_sets(model.extract(args), args.load_iteration, pipeline.extract(args), current_inpaint_round, cluster_idx, semantic_mask_bit, cluster_path)