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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from torchvision import transforms
from torchvision.transforms import InterpolationMode

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, only_pose = False):
    orig_w, orig_h = cam_info.image.size

    ret_scale = None
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        if resolution != (orig_w, orig_h):
            ret_scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        if resolution != (orig_w, orig_h):
            ret_scale = scale

    if only_pose:
        resized_image_rgb = torch.empty((3, resolution[1], resolution[0]), device=args.data_device)
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if only_pose:
        gt_semantic = torch.empty((1, resolution[1], resolution[0]), device=args.data_device)
    else:
        resize_transform = transforms.Resize((resolution[1], resolution[0]), interpolation=InterpolationMode.NEAREST)
        gt_semantic = resize_transform(cam_info.semantic_map.unsqueeze(0))

    ret_camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                        FoVx=cam_info.FovX, FoVy=cam_info.FovY, K=cam_info.K,
                        image=gt_image, gt_alpha_mask=loaded_mask,
                        semantic_map = gt_semantic.squeeze(0),
                        resize_scale = ret_scale,
                        image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    return ret_camera, ret_scale

def cameraList_from_camInfos(cam_infos, resolution_scale, args, only_pose=False):
    camera_list = []
    reso_list = []

    for id, c in enumerate(cam_infos):
        cam, reso_scale = loadCam(args, id, c, resolution_scale, only_pose)
        camera_list.append(cam)
        reso_list.append(reso_scale)

    return camera_list, reso_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : int(camera.width),
        'height' : int(camera.height),
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : float(fov2focal(camera.FovY, camera.height)),
        'fx' : float(fov2focal(camera.FovX, camera.width))
    }
    return camera_entry
