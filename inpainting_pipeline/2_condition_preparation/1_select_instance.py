
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
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.system_utils import searchForMaxIteration, searchForMaxInpaintRound
from arguments import ModelParams, get_combined_args
from scene import GaussianModel


def generate_pcd_valid_mask(
        dataset: ModelParams,
        load_iteration: int,
        instance_id: int,
        cluster_idx,
        solid_cluster_mask,
        instance_semantic: int,
        current_inpaint_round: int,
        if_all: bool,
):
    """
        After we have the cluster mask for each gaussian point, we can try to remove the specific instance with the selected instance_id .
        Or we can set if_all to True to remove all the instances.
    """
    gaussians = GaussianModel(dataset.sh_degree)
    loaded_iter = None
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

    semantic_mask = gaussians.get_semantic_splatting_mask(instance_semantic)

    if if_all:
        removed_pcd_mask = solid_cluster_mask
    else:
        active_cluster_mask = torch.zeros_like(cluster_idx).to(cluster_idx).bool()
        for i in instance_id:
            active_cluster_mask = active_cluster_mask + (cluster_idx == i)
        temp = semantic_mask.clone()
        temp[temp == True] = active_cluster_mask
        removed_pcd_mask = temp

    # removed_pcd_mask saved the mask for each gaussian point, which is True if the point is to be removed.
    workspace_path = os.path.join(dataset.model_path, "instance_workspace_{}".format(current_inpaint_round))
    torch.save(removed_pcd_mask, os.path.join(workspace_path, "removed_pcd_mask.pt"))
    gaussians.prune_splatting_with_mask(~removed_pcd_mask)
    gaussians.save_semantic_ply(os.path.join(workspace_path, "removed_pcd.ply"))

if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--instance_id", default=-1, type=int, nargs='+')
    # This pipeline may be run multiple times if necessary, so we need to keep track of the inpaint round.
    parser.add_argument("--current_inpaint_round", default=-1, type=int)
    parser.add_argument("--cluster_path", default="", type=str)
    parser.add_argument("--solid_cluster_mask", default="", type=str)
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--all", action="store_true")
    args = get_combined_args(parser)

    print(args.instance_id)
    if not args.all:
        assert args.instance_id != -1

    current_inpaint_round = None
    if args.current_inpaint_round == -1:
        current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path)
    else:
        current_inpaint_round = args.current_inpaint_round

    assert current_inpaint_round is not None

    current_inpaint_workspace = os.path.join(model.extract(args).model_path,
                                             "instance_workspace_{}".format(current_inpaint_round))

    if args.cluster_path:
        cluster_path = args.cluster_path
    else:
        cluster_path = os.path.join(current_inpaint_workspace, "cluster_idx.pt")

    (cluster_idx, semantic_mask_bit) = torch.load(cluster_path)
    cluster_idx = cluster_idx.cuda()

    if args.solid_cluster_mask:
        solid_cluster_mask = args.solid_cluster_mask
    else:
        solid_cluster_mask = os.path.join(current_inpaint_workspace, "solid_cluster_mask.pt")

    solid_cluster_mask = torch.load(solid_cluster_mask).cuda()

    generate_pcd_valid_mask(
        model.extract(args),
        args.load_iteration,
        args.instance_id,
        cluster_idx,
        solid_cluster_mask,
        semantic_mask_bit,
        current_inpaint_round,
        args.all,
    )
