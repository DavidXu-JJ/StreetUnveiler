
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
from os import makedirs
from random import randint
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torchvision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from scene.env_map import SkyModel
from utils.loss_utils import l1_loss
from utils.semantic_utils import semantic_prob_to_rgb
from utils.system_utils import searchForMaxIteration, mkdir_p, searchForMaxInpaintRound
from arguments import ModelParams, PipelineParams, ReOptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from scene.mask_gaussian import MaskGaussianModel
from gaussian_renderer import render, render_with_mask, render_semantic
from diffusers.utils import load_image
from inpainting_pipeline.utils import dilate_mask


def refine(
        dataset: ModelParams,
        opt: ReOptimizationParams,
        pipeline: PipelineParams,
        gaussians: MaskGaussianModel,
        sky_model: SkyModel,
        editable_pcd_mask,
        **kwargs,
):
    # Extract parameters from kwargs with default values if they are not provided
    current_inpaint_round = kwargs.get('current_inpaint_round', 0)  # Assuming 0 as a default value
    key_frame_list = kwargs.get('key_frame_list', [])
    mask_png_files = kwargs.get('mask_png_files', [])
    mask_npy_files = kwargs.get('mask_npy_files', [])
    rgb_images = kwargs.get('rgb_images', [])
    rgb_tensors = kwargs.get('rgb_tensors', [])
    instance_workspace = kwargs.get('instance_workspace_path', '')
    testing_iterations = kwargs.get('testing_iterations', [])
    inpaint_left_refill = kwargs.get('inpaint_left_refill', None)
    inpaint_zits = kwargs.get('inpaint_zits', None)

    inpainted_rgb_dict = dict()

    mask_np_list = [np.load(f) for f in mask_npy_files]
    masks_torch_list = [torch.from_numpy(mask).to(torch.bool).cuda() for mask in mask_np_list]

    middle_inpaint = os.path.join(instance_workspace, "middle_inpaint")
    makedirs(middle_inpaint, exist_ok=True)
    middle_render_dir = os.path.join(instance_workspace, "middle_render")
    makedirs(middle_render_dir, exist_ok=True)
    middle_mask_dir = os.path.join(instance_workspace, "middle_mask")
    makedirs(middle_mask_dir, exist_ok=True)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    tb_writer = prepare_output_and_logger(dataset, current_inpaint_round)

    key_frame_list = sorted(key_frame_list)
    if (scene.camera_frame_dict['front_end']) not in key_frame_list:
        key_frame_list.append(scene.camera_frame_dict['front_end'])

    count = 0
    next_editable_pcd_mask = editable_pcd_mask
    already_in_frame_mask = torch.zeros_like(next_editable_pcd_mask).cuda().bool()
    candidate_frames = []

    first_inpaint_key = True
    last_inpaint_image = None
    for i, last_frame_id in zip(reversed(key_frame_list[:-1]), reversed(key_frame_list[1:])):
        with torch.no_grad():
            current_image_size = load_image(mask_png_files[i]).size

            in_frame_mask = scene.getPcdInTrainFrame(gaussians.get_xyz, i)[1]
            trainable_mask = in_frame_mask * next_editable_pcd_mask
            next_editable_pcd_mask = next_editable_pcd_mask * (~in_frame_mask)

            viewpoint_cam = scene.getTrainCameras()[i]

            last_inframe_in_current_frame_render = render_with_mask(viewpoint_cam, gaussians, pipeline, background, ~already_in_frame_mask)["render"]
            current_frame_render = render(viewpoint_cam, gaussians, pipeline, background)["render"]
            """
            # Just place here for better understanding of what is each mask
            torchvision.utils.save_image(last_inframe_in_current_frame_render, os.path.join(middle_mask_dir, '{0:05d}_rgb1'.format(i) + ".png"))
            torchvision.utils.save_image(current_frame_render, os.path.join(middle_mask_dir, '{0:05d}_rgb2'.format(i) + ".png"))
            """

            already_in_frame_mask = already_in_frame_mask + in_frame_mask

            # visualize the current scene after removal
            render_pkg = render_with_mask(viewpoint_cam, gaussians, pipeline, background, ~trainable_mask)
            current_mask_torch = masks_torch_list[i]
            current_mask_torch = dilate_mask(current_mask_torch, 10)
            generated_mask_path = os.path.join(middle_mask_dir, '{0:05d}'.format(i) + ".png")
            torchvision.utils.save_image(current_mask_torch[None, ...].float().clamp(0, 1), generated_mask_path)

            # RGB
            render_image = render_pkg["render"]
            original_alpha = render_pkg['rend_alpha']
            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            render_image = render_image + sky_image * (1 - original_alpha)
            current_image_path = os.path.join(middle_render_dir, '{0:05d}'.format(i) + ".png")
            torchvision.utils.save_image(render_image, current_image_path)

            if first_inpaint_key == True:
                # Directly inpainting with zits for the first key frame
                current_mask_torch = masks_torch_list[i]
                current_mask_torch = dilate_mask(current_mask_torch, 15)
                generated_mask_path = os.path.join(middle_mask_dir, '{0:05d}'.format(i) + ".png")
                torchvision.utils.save_image(current_mask_torch[None, ...].float().clamp(0, 1), generated_mask_path)

                current_zits_inpainted_path = os.path.join(middle_inpaint, '{0:05d}_zits'.format(i) + ".png")
                inpaint_zits.inpaint(
                    current_image_path,
                    generated_mask_path,
                    current_zits_inpainted_path,
                )
                inpainted_rgb = load_image(current_zits_inpainted_path).resize(current_image_size, resample=Image.Resampling.BICUBIC)
                last_inpaint_image = inpainted_rgb
                first_inpaint_key = False

            else:
                # First inpaint with zits
                preprocess_zits_inpainted_path = os.path.join(middle_inpaint, '{0:05d}_preprocess_zits'.format(i) + ".png")
                preprocess_zits_mask = dilate_mask(masks_torch_list[i], 10)
                preprocess_zits_mask_path = os.path.join(middle_mask_dir, '{0:05d}_preprocess_zits'.format(i) + ".png")
                torchvision.utils.save_image(preprocess_zits_mask[None, ...].float().clamp(0, 1), preprocess_zits_mask_path)
                inpaint_zits.inpaint(
                    current_image_path,
                    preprocess_zits_mask_path,
                    preprocess_zits_inpainted_path,
                )
                preprocess_image = load_image(preprocess_zits_inpainted_path).resize(current_image_size, resample=Image.Resampling.BICUBIC)

                # Then inpaint the scene visible in future frame with left refill to maintain consistency
                eps = 2e-2
                current_diffuse_path = os.path.join(middle_inpaint, '{0:05d}_refill'.format(i) + ".png")
                refill_mask = ~((last_inframe_in_current_frame_render - current_frame_render).abs().sum(0) < eps) * masks_torch_list[i]
                refill_mask = dilate_mask(refill_mask, 10)
                refill_mask_path = os.path.join(middle_mask_dir, '{0:05d}_refill'.format(i) + ".png")
                torchvision.utils.save_image(refill_mask[None, ...].float().clamp(0, 1), refill_mask_path)
                inpainted_rgb = inpaint_left_refill.predict(
                    preprocess_image,
                    load_image(refill_mask_path),
                    last_inpaint_image,
                    ddim_steps=20,
                )[0].resize(current_image_size, resample=Image.Resampling.BICUBIC)
                inpainted_rgb.save(current_diffuse_path)
                last_inpaint_image = inpainted_rgb


            # Record for reoptimization
            inpainted_rgb_dict[i] = torch.from_numpy(np.array(inpainted_rgb)).cuda().permute(2, 0, 1) / 255.

            forward_inpaint_frames_num = last_frame_id - i - 1
            forward_inpaint_frames = [(i + 1 + j) for j in range(forward_inpaint_frames_num)]

            for frame_id in forward_inpaint_frames:
                # Inpaint the middle frames with left refill to maintain consistency
                current_image_size = load_image(mask_png_files[frame_id]).size
                reference_image = inpainted_rgb

                ref_inpaint_mask = dilate_mask(masks_torch_list[frame_id], 10)
                ref_inpaint_mask_png_path = os.path.join(middle_mask_dir, '{0:05d}'.format(frame_id) + ".png")
                torchvision.utils.save_image(ref_inpaint_mask[None, ...].float().clamp(0, 1), ref_inpaint_mask_png_path)

                ref_inpainted_rgb = inpaint_left_refill.predict(
                    load_image(rgb_images[frame_id]),
                    load_image(ref_inpaint_mask_png_path),
                    reference_image,
                    ddim_steps=20,
                )[0].resize(current_image_size, resample=Image.Resampling.BICUBIC)

                current_diffuse_path = os.path.join(middle_inpaint, '{0:05d}'.format(frame_id) + ".png")
                ref_inpainted_rgb.save(current_diffuse_path)

                inpainted_rgb_dict[frame_id] = torch.from_numpy(np.array(ref_inpainted_rgb)).cuda().permute(2, 0, 1) / 255.


        # Some reoptimization
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(0, opt.iterations), desc="Refining progress")

        viewpoint_stack = None
        candidate_frames += [(i + j) for j in range(forward_inpaint_frames_num + 1)]
        candidate_mask_dict = {}
        for frame_id in candidate_frames:
            candidate_mask_dict[frame_id] = masks_torch_list[frame_id]

        for iteration in range(1, opt.iterations + 1):
            gaussians.update_learning_rate(iteration)

            if not viewpoint_stack:
                viewpoint_stack = candidate_frames.copy()

            select_frame_id = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            viewpoint_cam = scene.getTrainCameras()[select_frame_id]

            sky_image = sky_model.render_with_camera(viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.K, viewpoint_cam.c2w)
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
            original_alpha = render_pkg['rend_alpha']
            render_image, render_depth = render_pkg["render"], render_pkg["surf_depth"]
            render_image = render_image + sky_image * (1 - original_alpha)

            loss_dict = {}

            current_supervision_rgb = inpainted_rgb_dict[select_frame_id]
            masked_render_image = torch.where(
                candidate_mask_dict[select_frame_id],
                render_image, torch.zeros_like(render_image).cuda()
            )
            masked_inpainted_image = torch.where(
                candidate_mask_dict[select_frame_id],
                current_supervision_rgb, torch.zeros_like(current_supervision_rgb).cuda()
            )

            unmasked_render_image = torch.where(
                ~candidate_mask_dict[select_frame_id],
                render_image, torch.zeros_like(render_image).cuda()
            )
            unmasked_gt_image = torch.where(
                ~candidate_mask_dict[select_frame_id],
                rgb_tensors[select_frame_id], torch.zeros_like(current_supervision_rgb).cuda()
            )

            Ll1 = l1_loss(masked_render_image, masked_inpainted_image) + l1_loss(unmasked_render_image, unmasked_gt_image)
            Ldist = opt.lambda_dist * render_pkg["rend_dist"].mean()
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            Lnormal = opt.lambda_normal * (normal_error).mean()


            loss = Ll1 + Ldist + Lnormal
            loss_dict['l1'] = Ll1
            loss_dict['ldist'] = Ldist
            loss_dict['lnormal'] = Lnormal

            loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                training_report(tb_writer, opt.iterations * count + iteration, loss_dict, testing_iterations, scene,
                                gaussians, render, (pipeline, background), sky_model)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

        count += 1

    checkpoint_path = os.path.join(instance_workspace, "checkpoint")
    mkdir_p(checkpoint_path)
    gaussians.save_ply(os.path.join(checkpoint_path, "point_cloud.ply"))
    gaussians.save_semantic_ply(os.path.join(checkpoint_path, "semantic_point_cloud.ply"))

    return gaussians, kwargs

def prepare_output_and_logger(args, current_inpaint_round):
    inpaint_dir = os.path.join(args.model_path, "instance_workspace_{}".format(current_inpaint_round))

    # Set up output folder
    print("Output folder: {}".format(inpaint_dir))
    os.makedirs(inpaint_dir, exist_ok=True)
    with open(os.path.join(inpaint_dir, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(inpaint_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss_dict, testing_iterations, scene: Scene, maskgaussians: MaskGaussianModel, renderFunc, renderArgs, sky_model):
    if tb_writer:
        for key, value in loss_dict.items():
            tb_writer.add_scalar('inpaint_loss_patches/{}_loss'.format(key), value.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           [100, 120, 130, 140, 160]]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, maskgaussians, *renderArgs)
                    env_image = sky_model.render_with_camera(viewpoint.image_height, viewpoint.image_width, viewpoint.K, viewpoint.c2w)
                    rend_alpha = render_pkg['rend_alpha']
                    image = torch.clamp(render_pkg["render"] + (1.0 - rend_alpha) * env_image, 0.0, 1.0)
                    disparity = torch.clamp(1.0 / render_pkg["surf_depth"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    render_semantics = render_semantic(viewpoint, maskgaussians, *renderArgs)["render_semantics"]
                    semantic_rgb = semantic_prob_to_rgb(render_semantics) / 255.
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/disparity".format(viewpoint.image_name),
                                             disparity[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/semantic".format(viewpoint.image_name),
                                             semantic_rgb[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                             rend_alpha[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/sky".format(viewpoint.image_name),
                                             env_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def prepare_basic_kwargs(args, model):
    # valid frame
    if args.load_iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model.extract(args).model_path, "checkpoint"))
    else:
        loaded_iter = args.load_iteration

    current_inpaint_round = None
    if args.current_inpaint_round == -1:
        current_inpaint_round = searchForMaxInpaintRound(model.extract(args).model_path)
    else:
        current_inpaint_round = args.current_inpaint_round

    assert current_inpaint_round is not None

    instance_workspace_path = os.path.join(model.extract(args).model_path, "instance_workspace_{}".format(current_inpaint_round))

    return_dict = {
        'loaded_iter': loaded_iter,
        'current_inpaint_round': current_inpaint_round,
        'instance_workspace_path': instance_workspace_path,
    }

    return return_dict

def prepare_refine_kwargs(valid_frame_list, key_frame_list, args, model, previous_dict=None):

    if previous_dict is None:
        return_dict = prepare_basic_kwargs(args, model)
    else:
        return_dict = previous_dict

    instance_workspace_path = return_dict['instance_workspace_path']

    # diffusion inpaint mask
    if args.mask_inpaint_path:
        mask_inpaint_path = args.mask_inpaint_path
    else:
        mask_inpaint_path = os.path.join(instance_workspace_path, "mask_inpaint")

    mask_png_files = []
    for frame_id in valid_frame_list:
        mask_png_files.append(os.path.join(mask_inpaint_path, '{0:05d}'.format(frame_id) + ".png"))
    mask_npy_files = []
    for frame_id in valid_frame_list:
        mask_npy_files.append(os.path.join(mask_inpaint_path, '{0:05d}'.format(frame_id) + ".npy"))

    # diffusion inpainted rgb
    if args.inpaint_image_path:
        inpaint_image_path = args.inpaint_image_path
    else:
        inpaint_image_path = os.path.join(instance_workspace_path, "inpainted_rgb")

    image_files = []
    for frame_id in valid_frame_list:
        image_files.append(os.path.join(inpaint_image_path, '{0:05d}'.format(frame_id) + ".png"))

    removed_image_tensors = []
    for frame_id in valid_frame_list:
        removed_image_tensors.append(torch.from_numpy(np.array(load_image(image_files[frame_id])) / 255.).float().permute(2,0,1).cuda())


    return_dict['key_frame_list'] = key_frame_list
    return_dict['mask_png_files'] = mask_png_files
    return_dict['mask_npy_files'] = mask_npy_files
    return_dict['rgb_images'] = image_files
    return_dict['rgb_tensors'] = removed_image_tensors
    return_dict['instance_workspace_path'] = instance_workspace_path

    return return_dict

def prepare_gaussians_and_scene(
        dataset: ModelParams,
        **kwargs,
):
    sky_model = SkyModel()
    sky_model.eval()
    loaded_iter = kwargs.get('loaded_iter', 0)  # Assuming 0 as a default value
    current_inpaint_round = kwargs.get('current_inpaint_round', 0)  # Assuming 0 as a default value

    gaussians = GaussianModel(dataset.sh_degree)
    if current_inpaint_round > 0:
        last_inpaint_checkpoint = os.path.join(
            dataset.model_path,
            "instance_workspace_{}".format(current_inpaint_round - 1),
            "checkpoint"
        )

        scene = Scene(
            dataset, gaussians, sky_model, load_iteration=loaded_iter, shuffle=False,
            only_pose=True,
            splatting_ply_path=os.path.join(last_inpaint_checkpoint, "point_cloud.ply"),
        )

        (model_params, first_iter) = torch.load(
            os.path.join(last_inpaint_checkpoint, "splatting.pt")
        )
        gaussians.restore(model_params, ReOptimizationParams(None))
    else:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "checkpoint"))

        scene = Scene(
            dataset, gaussians, sky_model, load_iteration=loaded_iter, shuffle=False,
            only_pose=True,
        )

        (model_params, first_iter) = torch.load(
            os.path.join(dataset.model_path, "checkpoint", "iteration_{}".format(loaded_iter), "splatting.pt")
        )
        gaussians.restore(model_params, ReOptimizationParams(None))

    return gaussians, scene, sky_model


def prepare_mask_gaussians(
        dataset: ModelParams,
        opt: ReOptimizationParams,
        gaussians: GaussianModel,
        removed_pcd_mask,
        trainable_pcd_mask,
):
    mask_gaussians = MaskGaussianModel(dataset.sh_degree)
    mask_gaussians.from_gaussian_model(gaussians)
    mask_gaussians.set_mask(trainable_pcd_mask * (~removed_pcd_mask))
    mask_gaussians.training_setup(opt)
    mask_gaussians.prune_points_with_mask(removed_pcd_mask)

    return mask_gaussians

def prepare_inpaint_kwargs(previous_dict):
    return_dict = previous_dict

    from utils.left_refill_utils import LeftRefillGuidance
    return_dict['inpaint_left_refill'] = LeftRefillGuidance()
    from utils.zits_utils import ZitsGuidance
    return_dict['inpaint_zits'] = ZitsGuidance()

    return return_dict


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = ReOptimizationParams(parser)
    parser.add_argument("--front_key_frames", nargs="+", type=int, required=True)
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--editable_pcd_mask_path", default="", type=str)
    parser.add_argument("--trainable_pcd_mask_path", default="", type=str)
    parser.add_argument("--mask_inpaint_path", default="", type=str)
    parser.add_argument("--inpaint_image_path", default="", type=str)
    parser.add_argument("--current_inpaint_round", default=-1, type=int)
    args = get_combined_args(parser)

    params_dict = prepare_basic_kwargs(args, model)
    gaussians, scene, sky_model = prepare_gaussians_and_scene(model.extract(args), **params_dict)

    valid_frame_list = [i for i in range(scene.camera_frame_dict['front_start'], scene.camera_frame_dict['front_end'])]
    refine_kwargs = prepare_refine_kwargs(valid_frame_list, args.front_key_frames, args, model, params_dict)
    inpaint_kwargs = prepare_inpaint_kwargs(refine_kwargs)

    # editing pcd
    if args.editable_pcd_mask_path:
        editable_pcd_mask_path = args.editable_pcd_mask_path
    else:
        editable_pcd_mask_path = os.path.join(params_dict['instance_workspace_path'], "editable_pcd_mask.pt")

    editable_pcd_mask = torch.load(editable_pcd_mask_path).cuda()

    # removed_pcd_mask refers to the removed gaussian points
    removed_pcd_mask_path = os.path.join(params_dict['instance_workspace_path'], "removed_pcd_mask.pt")
    removed_pcd_mask = torch.load(removed_pcd_mask_path).cuda()

    # trainable gaussian point mask for MaskGaussianModel
    if args.trainable_pcd_mask_path:
        trainable_pcd_mask_path = args.trainable_pcd_mask_path
    else:
        trainable_pcd_mask_path = os.path.join(params_dict['instance_workspace_path'], "trainable_pcd_mask.pt")

    trainable_pcd_mask = torch.load(trainable_pcd_mask_path).cuda()

    mask_gaussians = prepare_mask_gaussians(
        model.extract(args),
        op.extract(args),
        gaussians,
        removed_pcd_mask,
        trainable_pcd_mask,
    )

    del gaussians

    mask_gaussians, inpaint_kwargs = refine(
        model.extract(args),
        op.extract(args),
        pipeline.extract(args),
        mask_gaussians,
        sky_model,
        editable_pcd_mask[~removed_pcd_mask],
        **inpaint_kwargs,
    )