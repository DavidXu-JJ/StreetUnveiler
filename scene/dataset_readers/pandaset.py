
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

import numpy as np
import torch
from pandaset import DataSet as pandaDataSet
from pyquaternion import Quaternion
from superpose3d import Superpose3D

from utils.pcd_utils import SemanticPointCloud
from utils.semantic_utils import cityscapes2concerned, concerned_classes_list, concerned_classes_ind_map
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers.basic_utils import CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly, storeSemanticPly, to_homo_np
from scene.dataset_readers.colmap import readColmapCameras
from scene.dataset_readers.projection_utils import getCullMaskPointCloudInFrame, getCertainSemanticMask

def generatePandasetSemanticLidarPcd(
        panda_scene,
        colmap_path: str,
        downsample_lidar_voxel_size: float = 0.1,
):
    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    consider_camera = ['front_camera', 'front_left_camera', 'front_right_camera']

    lidar_ply_path = os.path.join(colmap_path, "lidar_points3D_{}.ply".format(downsample_lidar_voxel_size))
    semantic_color_ply_path = os.path.join(colmap_path, "lidar_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    ply_semantic_index_path = os.path.join(colmap_path, "lidar_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    pcd_list = []
    rgb_list = []
    semantic_list = []
    count = 0
    sl = slice(None, None, 1)
    lidars = panda_scene.lidar[sl]
    for camera_idx, camera in enumerate(consider_camera):
        current_camera = panda_scene.camera[camera]
        poses = current_camera.poses[sl]
        intrinsic = current_camera.intrinsics
        intr = np.zeros((3, 4))
        intr[0, 0] = intrinsic.fx
        intr[1, 1] = intrinsic.fy
        intr[0, 2] = intrinsic.cx
        intr[1, 2] = intrinsic.cy
        intr[2, 2] = 1

        frame_num = len(current_camera.poses)
        lidar_frame_num = len(lidars)
        assert frame_num == lidar_frame_num
        for idx in range(frame_num):
            current_pose = poses[idx]

            q = current_pose['heading']
            w, x, y, z = q['w'], q['x'], q['y'], q['z']
            rotation_matrix = Quaternion(w, x, y, z).rotation_matrix

            data_dict = current_pose['position']
            current_position = np.array([[data_dict['x'], data_dict['y'], data_dict['z']]])
            current_extr = np.eye(4)
            current_extr[:3, :3] = rotation_matrix
            current_extr[:3, 3] = current_position[0]

            w2c = np.linalg.inv(current_extr)

            current_lidar_pos = lidars[idx].values[:, :3]
            current_lidar_pos_homo = to_homo_np(current_lidar_pos)

            width, height = current_camera[0].size
            w, h = width, height
            valid_mask, valid_pix = getCullMaskPointCloudInFrame(h, w, current_lidar_pos_homo, w2c, intr)

            xyz = current_lidar_pos[valid_mask]
            current_image = np.array(current_camera[idx])
            rgb = current_image[valid_pix[..., 1], valid_pix[..., 0]]

            mask_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(count))
            semantic_map = np.load(mask_path)['arr_0']
            semantic_map = cityscapes2concerned(semantic_map)

            certain_semantic_mask = getCertainSemanticMask(semantic_map, valid_pix)

            xyz = xyz[certain_semantic_mask]
            rgb = rgb[certain_semantic_mask]
            semantic = semantic_map[valid_pix[..., 1], valid_pix[..., 0]][certain_semantic_mask]

            pcd_list.append(xyz)
            rgb_list.append(rgb)
            semantic_list.append(semantic)
            count += 1

    xyz_np = np.concatenate(pcd_list, axis=0)
    rgb_np = np.concatenate(rgb_list, axis=0).astype(float)
    semantic_np = np.concatenate(semantic_list, axis=0)

    semantic_pcd = SemanticPointCloud(points=xyz_np,
                                      colors=rgb_np,
                                      semantics=semantic_np,
                                      semantics_dict=concerned_classes_ind_map)
    semantic_pcd = semantic_pcd.voxel_down_sample(downsample_lidar_voxel_size)

    output_xyz = semantic_pcd.points.cpu().numpy()
    output_rgb = semantic_pcd.colors.to(torch.uint8).cpu().numpy()
    output_semantic = semantic_pcd.semantics.to(torch.uint8).cpu().numpy()
    storePly(lidar_ply_path, output_xyz, output_rgb)
    storeSemanticPly(semantic_color_ply_path, output_xyz, output_semantic)
    torch.save(semantic_pcd.semantics.to(torch.uint8).cpu(), ply_semantic_index_path)

    return

def getPandasetCameraPosition(
        panda_scene
):
    consider_camera = ['front_camera', 'front_left_camera', 'front_right_camera']

    sl = slice(None, None, 1)
    camera_position = []
    for camera_idx, camera in enumerate(consider_camera):
        current_camera = panda_scene.camera[camera]
        poses = current_camera.poses[sl]

        frame_num = len(poses)
        for i in range(frame_num):
            current_pose = poses[i]
            data_dict = current_pose['position']
            current_position = np.array([[data_dict['x'], data_dict['y'], data_dict['z']]]) # [1, 3]
            camera_position.append(current_position)

    camera_position = np.vstack(camera_position)

    return camera_position


def getPandaset2Colmap(
        panda_scene,
        colmap_path: str,
        images: str,
):
    # colmap
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(colmap_path, reading_dir), ignore_image=True)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    colmap_translation_list = []
    for info in cam_infos:
        colmap_w2c = np.eye(4)
        colmap_w2c[:3, :3] = np.transpose(info.R)
        colmap_w2c[:3, 3] = info.T
        colmap_c2w = np.linalg.inv(colmap_w2c)
        colmap_translation_list.append(colmap_c2w[:3, 3])

    colmap_translation_list = np.vstack(colmap_translation_list)

    pandaset_cam_positions = getPandasetCameraPosition(panda_scene)

    rmsd, R_ij, T_i, c = Superpose3D(colmap_translation_list, pandaset_cam_positions, None, True, False)

    NuScenes2Colmap = np.eye(4)
    NuScenes2Colmap[:3, :3] = np.array(R_ij) * c
    NuScenes2Colmap[:3, 3] = np.array(T_i)

    return NuScenes2Colmap


def getPandasetColmapSemanticPcd(
        panda_scene,
        colmap_path: str,
        images: str,
):

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

    pandaset2colmap = getPandaset2Colmap(panda_scene, colmap_path, images)
    colmap2pandaset = np.linalg.inv(pandaset2colmap)

    sfm_pcd = fetchPly(colmap_ply_path)
    sfm_pcd_rgb = sfm_pcd.colors.cpu().numpy()
    sfm_pcd_xyz = sfm_pcd.points.cpu().numpy()
    sfm_pcd_xyz_homo = np.concatenate([sfm_pcd_xyz, np.ones(sfm_pcd_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    sfm_pcd_xyz_homo = (colmap2pandaset @ sfm_pcd_xyz_homo.T).T

    sum_semantic = np.zeros((sfm_pcd_xyz.shape[0], len(concerned_classes_list)))
    counts = np.zeros(sfm_pcd_xyz.shape[0])[..., None]

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    consider_camera = ['front_camera', 'front_left_camera', 'front_right_camera']
    image_count = 0

    sl = slice(None, None, 1)
    for camera_idx, camera in enumerate(consider_camera):
        current_camera = panda_scene.camera[camera]
        poses = current_camera.poses[sl]
        intrinsic = current_camera.intrinsics
        intr = np.zeros((3, 4))
        intr[0, 0] = intrinsic.fx
        intr[1, 1] = intrinsic.fy
        intr[0, 2] = intrinsic.cx
        intr[1, 2] = intrinsic.cy
        intr[2, 2] = 1

        frame_num = len(current_camera.poses)
        for idx in range(frame_num):
            current_pose = poses[idx]

            q = current_pose['heading']
            w, x, y, z = q['w'], q['x'], q['y'], q['z']
            rotation_matrix = Quaternion(w, x, y, z).rotation_matrix

            data_dict = current_pose['position']
            current_position = np.array([[data_dict['x'], data_dict['y'], data_dict['z']]])
            current_extr = np.eye(4)
            current_extr[:3, :3] = rotation_matrix
            current_extr[:3, 3] = current_position[0]

            w2c = np.linalg.inv(current_extr)

            width, height = current_camera[0].size
            w, h = width, height
            valid_mask, valid_pix = getCullMaskPointCloudInFrame(h, w, sfm_pcd_xyz_homo, w2c, intr)

            mask_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(image_count))
            semantic_map = np.load(mask_path)['arr_0']
            semantic_map = cityscapes2concerned(semantic_map)

            select_semantic = semantic_map[valid_pix[..., 1], valid_pix[..., 0]]

            sum_semantic[valid_mask, select_semantic] += 1
            counts[valid_mask] += 1

    valid_mask = (counts > 0).squeeze(-1)

    sfm_pcd_xyz = sfm_pcd_xyz[valid_mask]
    sfm_pcd_rgb = sfm_pcd_rgb[valid_mask]

    semantic_tag = np.argmax(sum_semantic[valid_mask], axis=1)

    assert semantic_tag.shape[0] == sfm_pcd_xyz.shape[0]
    return sfm_pcd_xyz, sfm_pcd_rgb, semantic_tag


def getPandasetSemanticPcd(
        panda_scene,
        colmap_path: str,
        images: str,
        downsample_lidar_voxel_size: float = 0.1,
):
    lidar_ply_path = os.path.join(colmap_path, "lidar_points3D_{}.ply".format(downsample_lidar_voxel_size))
    semantic_color_ply_path = os.path.join(colmap_path, "lidar_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    ply_semantic_index_path = os.path.join(colmap_path, "lidar_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    if (
            not os.path.exists(lidar_ply_path)
            or not os.path.exists(semantic_color_ply_path)
            or not os.path.exists(ply_semantic_index_path)
    ):
        generatePandasetSemanticLidarPcd(panda_scene, colmap_path, downsample_lidar_voxel_size)

    lidar_pcd = fetchPly(lidar_ply_path)
    lidar_semantic = torch.load(ply_semantic_index_path)
    lidar_pcd.load_semantics(lidar_semantic)

    final_ply_path = os.path.join(colmap_path, "final_points3D_{}.ply".format(downsample_lidar_voxel_size))
    final_semantic_color_ply_path = os.path.join(colmap_path, "final_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    final_ply_semantic_index_path = os.path.join(colmap_path, "final_points3D_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    if (
            not os.path.exists(final_ply_path)
            or not os.path.exists(final_semantic_color_ply_path)
            or not os.path.exists(final_ply_semantic_index_path)
    ):
        sfm_xyz, sfm_rgb, sfm_semantic = getPandasetColmapSemanticPcd(panda_scene, colmap_path, images)
        sfm_pcd = SemanticPointCloud(points=sfm_xyz, colors=sfm_rgb, semantics=sfm_semantic)

        final_xyz = torch.cat([lidar_pcd.points, sfm_pcd.points], dim=0)
        final_rgb = torch.cat([lidar_pcd.colors, sfm_pcd.colors], dim=0) * 255.0
        final_semantic = torch.cat([lidar_pcd.semantics, sfm_pcd.semantics], dim=0)

        pandaset2colmap = getPandaset2Colmap(panda_scene, colmap_path, images)
        final_xyz_homo = torch.cat([final_xyz, torch.ones(final_xyz.shape[0], 1).to(final_xyz)], dim=-1)
        final_xyz_homo = (torch.from_numpy(pandaset2colmap).float().cuda() @ final_xyz_homo.T).T
        final_xyz = final_xyz_homo[..., :3] / final_xyz_homo[..., 3][..., None]

        storePly(final_ply_path, final_xyz.cpu().numpy(), final_rgb.cpu().numpy())
        storeSemanticPly(final_semantic_color_ply_path, final_xyz.cpu().numpy(), final_semantic.cpu().numpy())
        torch.save(final_semantic, final_ply_semantic_index_path)

    final_pcd = fetchPly(final_ply_path)
    final_pcd_semantic = torch.load(final_ply_semantic_index_path)
    final_pcd.load_semantics(final_pcd_semantic)

    return lidar_pcd, final_pcd, lidar_ply_path, final_ply_path


def readPandasetInfo(
        pandaset_path: str,
        images: str,
        colmap_path: str,
        eval: bool = False,
        llffhold: int = 8,
):
    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    panda = pandaDataSet(pandaset_path)
    scene_name = os.path.basename(colmap_path)
    panda_scene = panda[scene_name]
    panda_scene.load_camera()
    panda_scene.load_lidar()

    lidar_pcd, final_pcd, lidar_ply_path, final_ply_path = getPandasetSemanticPcd(panda_scene, colmap_path, images)

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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(colmap_path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    for i in range(len(cam_infos)):
        mask_file = os.path.join(colmap_path, "images_masks", idx_to_mask_filename(i))
        semantic_map = np.load(mask_file)['arr_0']
        semantic_map = torch.from_numpy(semantic_map).to(torch.int64)
        semantic_map = cityscapes2concerned(semantic_map)
        cam_infos[i] = cam_infos[i]._replace(semantic_map=semantic_map)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=final_pcd,
                           reference_cloud=lidar_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=final_ply_path,
                           reference_ply_path=lidar_ply_path,)

    camera_frame_dict = {}
    camera_frame_dict['front_start'] = 0
    front_camera = panda_scene.camera['front_camera']
    frame_num = len(front_camera.poses)
    camera_frame_dict['front_end'] = frame_num

    return scene_info, camera_frame_dict

