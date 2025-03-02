
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

import os
import pickle
from typing import NamedTuple, List, Optional, Tuple
from PIL import Image

import torch
import numpy as np

from utils.pcd_utils import SemanticPointCloud
from utils.semantic_utils import cityscapes2concerned, concerned_classes_list, concerned_classes_ind_map, WAYMO_CAMERAS, WAYMO_LIDARS
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers.basic_utils import CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly, storeSemanticPly
from scene.dataset_readers.colmap import readColmapCameras
from scene.dataset_readers.projection_utils import getCullMaskPointCloudInFrame, getCertainSemanticMask

from superpose3d import Superpose3D

WAYMO_CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT']
WAYMO_LIDARS = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']

def idx_to_frame_str(frame_index):
    return f'{frame_index:08d}'

def idx_to_camera_id(camera_index):
    return f'camera_{WAYMO_CAMERAS[camera_index]}'

def idx_to_mask_filename(frame_index, compress=True):
    ext = 'npz' if compress else 'npy'
    return f'{idx_to_frame_str(frame_index)}.{ext}'

def idx_to_lidar_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.npz'

def idx_to_lidar_id(lidar_index):
    return f'lidar_{WAYMO_LIDARS[lidar_index]}'

def idx_to_img_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.jpg'

def getWaymoCameraStartEndIdx(
        scenario: dict,
):
    camera_start_idx = {}
    camera_end_idx = {}
    count = 0
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_id = idx_to_camera_id(camera_idx)
        camera_start_idx[camera_str_id] = count
        count += scenario['observers'][camera_str_id]['n_frames']
        camera_end_idx[camera_str_id] = count

    return camera_start_idx, camera_end_idx

def getWaymoTranslationList(
        scenario: dict,
):
    sampled_points = []
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_idx = idx_to_camera_id(camera_idx)
        for frame_idx in range(scenario['observers'][camera_str_idx]['n_frames']):
            c2w = scenario['observers'][camera_str_idx]['data']['c2w'][frame_idx]
            sampled_points.append(c2w[:3, 3])

    return sampled_points

def getWaymoPointCloudSemanticFromImageAtCertainFrame(
        path: str,
        colmap_path: str,
        scenario: dict,
        xyz: np.ndarray,
        frame_idx: int,
) -> np.ndarray:

    counts = np.zeros(xyz.shape[0])[..., None]
    sum_rgbs = np.zeros_like(xyz)
    semantic_class = (-1 * np.ones((xyz.shape[0]))).astype(np.int32)    # [num_points]
    xyz_homo = np.concatenate([xyz, np.ones(xyz.shape[:-1])[:, np.newaxis]], axis=-1)

    head_idx = {}
    count = 0
    head_idx[WAYMO_CAMERAS[0]] = count
    for camera_idx in range(len(WAYMO_CAMERAS)-1):
        camera_str_idx = idx_to_camera_id(camera_idx)

        frame_nums = len(scenario['observers'][camera_str_idx]['data']['hw'])
        count += frame_nums
        head_idx[WAYMO_CAMERAS[camera_idx + 1]] = count


    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_idx = idx_to_camera_id(camera_idx)

        h, w = scenario['observers'][camera_str_idx]['data']['hw'][frame_idx]
        image_path = os.path.join(path, "images", camera_str_idx, idx_to_img_filename(frame_idx))
        with Image.open(image_path) as img:
            image = np.array(img)
        mask_file = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(frame_idx + head_idx[WAYMO_CAMERAS[camera_idx]]))
        # [h, w]
        semantic_map = np.load(mask_file)['arr_0']
        semantic_map = cityscapes2concerned(semantic_map)

        w2c = np.linalg.inv(scenario['observers'][camera_str_idx]['data']['c2w'][frame_idx])

        # [3, 4]
        intr = np.zeros((3, 4))
        intr[:3, :3] = scenario['observers'][camera_str_idx]['data']['intr'][frame_idx]

        mask, valid_pix = getCullMaskPointCloudInFrame(h, w, xyz_homo, w2c, intr)

        certain_mask = getCertainSemanticMask(semantic_map, valid_pix)

        final_certain_pix = valid_pix[certain_mask]
        semantic = semantic_map[final_certain_pix[..., 1], final_certain_pix[..., 0]]

        selected_color = image[final_certain_pix[..., 1], final_certain_pix[..., 0], :]

        sum_rgbs_valid = sum_rgbs[mask]
        sum_rgbs_valid[certain_mask] += selected_color
        sum_rgbs[mask] = sum_rgbs_valid

        semantic_class_valid = semantic_class[mask]
        semantic_class_valid[certain_mask] = semantic.astype(np.int32)
        semantic_class[mask] = semantic_class_valid

        counts_valid = counts[mask]
        counts_valid[certain_mask] += 1
        counts[mask] = counts_valid

    valid_mask = (counts > 0).squeeze(-1)

    xyz = xyz[valid_mask]
    semantic_class = semantic_class[valid_mask]
    rgb = sum_rgbs[valid_mask] / counts[valid_mask]

    assert semantic_class.shape[0] == xyz.shape[0]
    return xyz, rgb, semantic_class

def getWaymoPointCloudSemanticFromImage(
        colmap_path: str,
        scenario: dict,
        xyz: np.ndarray,
) -> np.ndarray:

    global_frame_idx = 0
    counts = np.zeros(xyz.shape[0])[..., None]
    sum_semantic = np.zeros((xyz.shape[0], len(concerned_classes_list)))
    xyz_homo = np.concatenate([xyz, np.ones(xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_idx = idx_to_camera_id(camera_idx)

        for frame_idx in range(scenario['observers'][camera_str_idx]['n_frames']):
            mask_file = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(global_frame_idx))
            # [h, w]
            semantic_map = np.load(mask_file)['arr_0']
            semantic_map = cityscapes2concerned(semantic_map)

            h, w = scenario['observers'][camera_str_idx]['data']['hw'][frame_idx]
            w2c = np.linalg.inv(scenario['observers'][camera_str_idx]['data']['c2w'][frame_idx])

            # [3, 4]
            intr = np.zeros((3, 4))
            intr[:3, :3] = scenario['observers'][camera_str_idx]['data']['intr'][frame_idx]

            mask, valid_pix = getCullMaskPointCloudInFrame(h, w, xyz_homo, w2c, intr)

            # [w, h] = [x, y]
            select_semantic = semantic_map[valid_pix[..., 1], valid_pix[..., 0]]

            sum_semantic[mask, select_semantic] += 1
            counts[mask] += 1
            global_frame_idx += 1

    valid_mask = (counts > 0).squeeze(-1)

    xyz = xyz[valid_mask]
    semantic_tag = np.argmax(sum_semantic[valid_mask], axis=1)

    assert semantic_tag.shape[0] == xyz.shape[0]
    return xyz, semantic_tag, valid_mask


def addWaymoLidarPointCloud(
        path: str,
        colmap_path: str,
        scenario: dict,
        downsample_voxel_size: float = 0.1,
):

    lidar_frame = []
    for lidar_idx in range(len(WAYMO_LIDARS)):
        lidar_str_id = idx_to_lidar_id(lidar_idx)
        lidar_frame.append(scenario['observers'][lidar_str_id]['n_frames'])

    all_xyz = []
    all_rgb = []
    all_semantic_class = []
    for frame_idx in range(max(lidar_frame)):
        # get all lidar points at this frame
        lidar_points_world = []
        for lidar_idx in range(len(WAYMO_LIDARS)):
            lidar_str_id = idx_to_lidar_id(lidar_idx)
            lidar_file_path = os.path.join(path, "lidars", lidar_str_id, idx_to_lidar_filename(frame_idx))
            if os.path.exists(lidar_file_path):
                arr_dict = np.load(lidar_file_path)
                rays_o = arr_dict['rays_o']
                rays_d = arr_dict['rays_d']
                ranges = arr_dict['ranges']
                valid_mask = (ranges > 0.)

                points = rays_o[valid_mask] + (ranges[valid_mask])[..., None] * rays_d[valid_mask]
                points_homo = np.concatenate([points, np.ones(points.shape[:-1])[:, np.newaxis]], axis=-1)
                l2w = scenario['observers'][lidar_str_id]['data']['l2w'][frame_idx]
                points_homo = (l2w @ points_homo.T).T
                lidar_points_world.append(points_homo[..., :3] / points_homo[..., 3][..., None])

        lidar_points_world = np.concatenate(lidar_points_world, axis=0)

        xyz, rgb, semantic_class = getWaymoPointCloudSemanticFromImageAtCertainFrame(
            path, colmap_path, scenario, lidar_points_world, frame_idx
        )

        # assert np.array_equal(xyz, _xyz)

        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_semantic_class.append(semantic_class)

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    all_semantic_class = np.concatenate(all_semantic_class, axis=0)

    print("All points:", all_xyz.shape)

    semantic_pcd = SemanticPointCloud(points=all_xyz,
                                      colors=all_rgb,
                                      semantics=all_semantic_class,
                                      semantics_dict=concerned_classes_ind_map)

    semantic_pcd = semantic_pcd.voxel_down_sample(downsample_voxel_size)
    print("Initialization point cloud downsampled to:", semantic_pcd.points.shape)
    return semantic_pcd.points, semantic_pcd.colors, semantic_pcd.semantics


def extractWaymoPcd(
        path: str,
        colmap_path: str,
        scenario: dict,
        waymo2colmap: Optional[np.ndarray] = None,
        downsample_lidar_voxel_size: float = 0.1,
):

    ply_path = os.path.join(path, "points3D_{}.ply".format(downsample_lidar_voxel_size))
    lidar_ply_path = os.path.join(path, "lidar_points3D_{}.ply".format(downsample_lidar_voxel_size))
    semantic_color_ply_path = os.path.join(path, "semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    ply_semantic_index_path = os.path.join(path, "points3D_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    final_ply_path = os.path.join(path, "final_points3D_{}.ply".format(downsample_lidar_voxel_size))
    final_semantic_color_ply_path = os.path.join(path, "final_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    final_ply_semantic_index_path = os.path.join(path, "final_points3D_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    if (
            not os.path.exists(ply_path)
            or not os.path.exists(lidar_ply_path)
            or not os.path.exists(ply_semantic_index_path)
            or not os.path.exists(semantic_color_ply_path)
            or not os.path.exists(final_ply_path)
            or not os.path.exists(final_semantic_color_ply_path)
            or not os.path.exists(final_ply_semantic_index_path)
    ):
        xyz_torch, rgb_torch, semantic_torch = addWaymoLidarPointCloud(
            path, colmap_path, scenario,
            downsample_lidar_voxel_size
        )

        xyz = xyz_torch.cpu().numpy()
        rgb = rgb_torch.cpu().numpy()

        lidar_xyz = xyz
        if waymo2colmap is not None:
            lidar_homo = np.concatenate([lidar_xyz, np.ones(lidar_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
            lidar_homo = (waymo2colmap @ lidar_homo.T).T
            lidar_xyz = lidar_homo[..., :3] / lidar_homo[..., 3][..., None]

        storePly(lidar_ply_path, lidar_xyz, rgb)

        semantic_torch_cpu = semantic_torch.cpu()
        torch.save(semantic_torch_cpu, ply_semantic_index_path)

        if waymo2colmap is not None:
            xyz_homo = np.concatenate([xyz, np.ones(xyz.shape[:-1])[:, np.newaxis]], axis=-1)
            # xyz = (waymo_w2c @ xyz.T).T
            # xyz = (colmap_c2w @ xyz.T).T
            xyz_homo = (waymo2colmap @ xyz_homo.T).T
            xyz = xyz_homo[..., :3] / xyz_homo[..., 3][..., None]

        storePly(ply_path, xyz, rgb)
        storeSemanticPly(semantic_color_ply_path, xyz, semantic_torch_cpu.numpy().astype(np.int32))

        previous_pcd = fetchPly(ply_path)
        xyz = previous_pcd.points.cpu().numpy()
        rgb = previous_pcd.colors.cpu().numpy()

        colmap_ply_path = os.path.join(colmap_path, "sparse/0/points3D.ply")
        bin_path = os.path.join(colmap_path, "sparse/0/points3D.bin")
        txt_path = os.path.join(colmap_path, "sparse/0/points3D.txt")
        if not os.path.exists(colmap_ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                _xyz, _rgb, _ = read_points3D_binary(bin_path)
            except:
                _xyz, _rgb, _ = read_points3D_text(txt_path)
            storePly(colmap_ply_path, _xyz, _rgb)

        colmap2waymo = np.linalg.inv(waymo2colmap)

        sfm_pcd = fetchPly(colmap_ply_path)
        sfm_pcd_xyz = sfm_pcd.points.cpu().numpy()
        sfm_pcd_xyz_homo = np.concatenate([sfm_pcd_xyz, np.ones(sfm_pcd_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
        sfm_pcd_xyz_homo = (colmap2waymo @ sfm_pcd_xyz_homo.T).T
        sfm_pcd_xyz = sfm_pcd_xyz_homo[..., :3] / sfm_pcd_xyz_homo[..., 3][..., None]

        sfm_pcd_rgb = sfm_pcd.colors.cpu().numpy()

        valid_sfm_xyz, valid_sfm_semantic, valid_sfm_mask = getWaymoPointCloudSemanticFromImage(colmap_path, scenario, sfm_pcd_xyz)
        valid_sfm_color = sfm_pcd_rgb[valid_sfm_mask]

        valid_sfm_xyz_homo = np.concatenate([valid_sfm_xyz, np.ones(valid_sfm_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
        valid_sfm_xyz_homo = (waymo2colmap @ valid_sfm_xyz_homo.T).T
        valid_sfm_xyz = valid_sfm_xyz_homo[..., :3] / valid_sfm_xyz_homo[..., 3][..., None]

        final_pcd_xyz = np.concatenate([xyz, valid_sfm_xyz], axis=0)
        final_pcd_rgb = np.concatenate([rgb, valid_sfm_color], axis=0) * 255.0
        final_pcd_semantic = torch.cat([semantic_torch_cpu, torch.from_numpy(valid_sfm_semantic)], dim=0)

        storePly(final_ply_path, final_pcd_xyz, final_pcd_rgb)
        storeSemanticPly(final_semantic_color_ply_path, final_pcd_xyz, final_pcd_semantic)
        torch.save(final_pcd_semantic, final_ply_semantic_index_path)

    final_pcd = fetchPly(final_ply_path)
    lidar_pcd = fetchPly(lidar_ply_path)
    final_pcd_semantic = torch.load(final_ply_semantic_index_path)
    final_pcd.load_semantics(final_pcd_semantic)

    return final_pcd, lidar_pcd, final_ply_path, lidar_ply_path, final_ply_semantic_index_path


def readWaymoInfo(
        path: str,
        images: str,
        colmap_path: Optional[str] = None,
        eval: bool = False,
        llffhold: int = 8,
):

    scenario_path = os.path.join(path, "scenario.pt")

    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f)

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(colmap_path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    for i in range(len(cam_infos)):
        mask_file = os.path.join(colmap_path, "images_masks", idx_to_mask_filename(i))
        semantic_map = np.load(mask_file)['arr_0']
        semantic_map = torch.from_numpy(semantic_map).to(torch.int64)
        semantic_map = cityscapes2concerned(semantic_map)
        cam_infos[i] = cam_infos[i]._replace(semantic_map=semantic_map)

    camera_start_idx, camera_end_idx = getWaymoCameraStartEndIdx(scenario)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        if camera_start_idx is not None:
            assert camera_end_idx is not None
            for camera_id in range(len(WAYMO_CAMERAS)):
                camera_str_id = idx_to_camera_id(camera_id)
                camera_start_idx[camera_str_id] -= (camera_start_idx[camera_str_id] - 1) // llffhold + 1
                camera_end_idx[camera_str_id] -= (camera_end_idx[camera_str_id] - 1) // llffhold + 1
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # List([3])
    # waymo camera position
    waymo_translation_list = getWaymoTranslationList(scenario)
    if eval:
        waymo_translation_list_with_eval_cam = [c for idx, c in enumerate(waymo_translation_list) if idx % llffhold != 0]
        waymo_translation_list = waymo_translation_list_with_eval_cam

    # colmap camera position
    colmap_translation_list = []
    for info in train_cam_infos:
        colmap_w2c = np.eye(4)
        colmap_w2c[:3, :3] = np.transpose(info.R)
        colmap_w2c[:3, 3] = info.T
        colmap_c2w = np.linalg.inv(colmap_w2c)
        colmap_translation_list.append(colmap_c2w[:3, 3])

    rmsd, R_ij, T_i, c = Superpose3D(np.array(colmap_translation_list), np.array(waymo_translation_list), None, True, False)

    waymo2colmap = np.eye(4)
    waymo2colmap[:3, :3] = np.array(R_ij) * c
    waymo2colmap[:3, 3] = np.array(T_i)

    pcd, reference_pcd, ply_path, lidar_ply_path, ply_semantic_index_path = extractWaymoPcd(path, colmap_path, scenario, waymo2colmap)

    scene_info = SceneInfo(point_cloud=pcd,
                           reference_cloud=reference_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           reference_ply_path=lidar_ply_path,)

    camera_frame_dict = {}
    camera_frame_dict['front_start'] = camera_start_idx[idx_to_camera_id(0)]
    camera_frame_dict['front_end'] = camera_end_idx[idx_to_camera_id(0)]
    camera_frame_dict['left_start'] = camera_start_idx[idx_to_camera_id(1)]
    camera_frame_dict['left_end'] = camera_end_idx[idx_to_camera_id(1)]
    camera_frame_dict['right_start'] = camera_start_idx[idx_to_camera_id(2)]
    camera_frame_dict['right_end'] = camera_end_idx[idx_to_camera_id(2)]
    return scene_info, camera_frame_dict


"""
# Some unused utility functions

def readWaymoCameras(
        path: str,
        scenario: dict,
        open_image: bool,
):

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_img_filename(frame_index):
        return f'{idx_to_frame_str(frame_index)}.jpg'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    def idx_to_camera_id(camera_index):
        return f'camera_{WAYMO_CAMERAS[camera_index]}'

    cam_infos = []
    count = 0
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_id = idx_to_camera_id(camera_idx)
        for frame_idx in range(scenario['observers'][camera_str_id]['n_frames']):
            h, w = scenario['observers'][camera_str_id]['data']['hw'][frame_idx]
            image_path = os.path.join(path, "images", camera_str_id, idx_to_img_filename(frame_idx))
            image_name = Path(idx_to_img_filename(frame_idx)).stem
            if open_image == False:
                image = None
            else:
                image = load_image(image_path)

            mask_file = os.path.join(path, "masks", camera_str_id, idx_to_mask_filename(frame_idx))
            # [h, w]
            semantic_map = np.load(mask_file)['arr_0']
            semantic_map = torch.from_numpy(semantic_map).to(torch.int64)
            semantic_map = cityscapes2concerned(semantic_map)

            # [3, 3]
            intr = scenario['observers'][camera_str_id]['data']['intr'][frame_idx]
            fx = intr[0, 0]
            fy = intr[1, 1]

            # FIXME: the [:2,3](the offset in pixel coords) of intr in waymo dataset is not strictly height/2 and width/2,
            # so this cam_info can't be used directly.
            fovx = focal2fov(fx, w)
            fovy = focal2fov(fy, h)

            c2w = scenario['observers'][camera_str_id]['data']['c2w'][frame_idx]
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3]) # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            cam_infos.append(
                CameraInfo(
                    uid = count,
                    R = R, T = T,
                    FovY = fovy, FovX = fovx,
                    image = image, image_path = image_path, image_name = image_name,
                    semantic_map = semantic_map,
                    width = w, height = h,
                    K = intr,
                )
            )
            count += 1

    return cam_infos

def getWaymoPointCloudColorFromImage(
        path: str,
        scenario: dict,
        xyz: np.ndarray,
) -> np.ndarray:

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_img_filename(frame_index):
        return f'{idx_to_frame_str(frame_index)}.jpg'

    def idx_to_camera_id(camera_index):
        return f'camera_{WAYMO_CAMERAS[camera_index]}'

    counts = np.zeros(xyz.shape[0])[..., None]
    sum_rgbs = np.zeros_like(xyz)
    xyz_homo = np.concatenate([xyz, np.ones(xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_idx = idx_to_camera_id(camera_idx)

        for frame_idx in range(scenario['observers'][camera_str_idx]['n_frames']):
            h, w = scenario['observers'][camera_str_idx]['data']['hw'][frame_idx]
            image_path = os.path.join(path, "images", camera_str_idx, idx_to_img_filename(frame_idx))
            with Image.open(image_path) as img:
                image = np.array(img)
            w2c = np.linalg.inv(scenario['observers'][camera_str_idx]['data']['c2w'][frame_idx])

            # [3, 4]
            intr = np.zeros((3, 4))
            intr[:3, :3] = scenario['observers'][camera_str_idx]['data']['intr'][frame_idx]

            mask, valid_pix = getCullMaskPointCloudInFrame(h, w, xyz_homo, w2c, intr)

            # [w, h] = [x, y]
            # TODO: change to interpolation of pixels
            selected_color = image[valid_pix[..., 1], valid_pix[..., 0], :]

            sum_rgbs[mask] += selected_color
            counts[mask] += 1

    valid_mask = (counts > 0).squeeze(-1)

    xyz = xyz[valid_mask]
    rgb = sum_rgbs[valid_mask] / counts[valid_mask]

    assert rgb.shape[0] == xyz.shape[0] and rgb.shape[1] == 3
    return xyz, rgb

def getWaymoPointCloudColorFromImageAtCertainFrame(
        path: str,
        scenario: dict,
        xyz: np.ndarray,
        frame_idx: int,
) -> np.ndarray:

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_img_filename(frame_index):
        return f'{idx_to_frame_str(frame_index)}.jpg'

    def idx_to_camera_id(camera_index):
        return f'camera_{WAYMO_CAMERAS[camera_index]}'

    counts = np.zeros(xyz.shape[0])[..., None]
    sum_rgbs = np.zeros_like(xyz)
    xyz_homo = np.concatenate([xyz, np.ones(xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    for camera_idx in range(len(WAYMO_CAMERAS)):
        camera_str_idx = idx_to_camera_id(camera_idx)

        h, w = scenario['observers'][camera_str_idx]['data']['hw'][frame_idx]
        image_path = os.path.join(path, "images", camera_str_idx, idx_to_img_filename(frame_idx))
        with Image.open(image_path) as img:
            image = np.array(img)
        w2c = np.linalg.inv(scenario['observers'][camera_str_idx]['data']['c2w'][frame_idx])

        # [3, 4]
        intr = np.zeros((3, 4))
        intr[:3, :3] = scenario['observers'][camera_str_idx]['data']['intr'][frame_idx]

        mask, valid_pix = getCullMaskPointCloudInFrame(h, w, xyz_homo, w2c, intr)

        # [w, h] = [x, y]
        # TODO: change to interpolation of pixels
        selected_color = image[valid_pix[..., 1], valid_pix[..., 0], :]

        sum_rgbs[mask] += selected_color
        counts[mask] += 1

    valid_mask = (counts > 0).squeeze(-1)

    xyz = xyz[valid_mask]
    rgb = sum_rgbs[valid_mask] / counts[valid_mask]

    assert rgb.shape[0] == xyz.shape[0] and rgb.shape[1] == 3
    return xyz, rgb

"""