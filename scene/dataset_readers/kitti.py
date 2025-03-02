
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
import re

import numpy as np
import pykitti
import torch

from superpose3d import Superpose3D

from utils.system_utils import count_png_files
from utils.pcd_utils import SemanticPointCloud
from utils.semantic_utils import cityscapes2concerned, concerned_classes_list, concerned_classes_ind_map
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers.basic_utils import CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly, storeSemanticPly, to_homo_np
from scene.dataset_readers.colmap import readColmapCameras
from scene.dataset_readers.projection_utils import getCullMaskPointCloudInFrame, getCertainSemanticMask


def getKitti2Colmap(
        kitti_data: pykitti.raw,
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

    cam_num = len(colmap_translation_list)
    colmap_translation_list = np.vstack(colmap_translation_list)
    colmap_translation_list = colmap_translation_list[: cam_num//2, :]

    # kitti
    kitti_translation_list = []
    point_velo = np.array([0, 0, 0, 1])
    velo2imu = np.linalg.inv(kitti_data.calib.T_velo_imu)
    kitti_translation_list = [(o.T_w_imu @ velo2imu @ point_velo) for o in kitti_data.oxts]

    kitti_translation_list = np.vstack(kitti_translation_list)
    kitti_translation_list = kitti_translation_list[..., :3] / kitti_translation_list[..., 3][..., None]

    rmsd, R_ij, T_i, c = Superpose3D(colmap_translation_list, kitti_translation_list, None, True, False)

    Kitti2Colmap = np.eye(4)
    Kitti2Colmap[:3, :3] = np.array(R_ij) * c
    Kitti2Colmap[:3, 3] = np.array(T_i)

    return Kitti2Colmap

def generateKittiSemanticLidarPcd(
    kitti_data: pykitti.raw,
    colmap_path: str,
    downsample_lidar_voxel_size: float,
    lidar_ply_path: str,
    semantic_color_ply_path: str,
    ply_semantic_index_path: str,
):
    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    w, h = kitti_data.get_cam2(0).size
    n_frames = len(kitti_data)

    pcd_iterator = kitti_data.velo
    rgb_iterator = kitti_data.rgb
    oxt_iterator = kitti_data.oxts

    image_id = 0
    velo2cam2 = kitti_data.calib.T_cam2_velo
    velo2cam3 = kitti_data.calib.T_cam3_velo
    intr_cam2 = kitti_data.calib.K_cam2
    intr_cam3 = kitti_data.calib.K_cam3

    imu2velo = kitti_data.calib.T_velo_imu
    velo2imu = np.linalg.inv(kitti_data.calib.T_velo_imu)

    pcd_xyz = []
    pcd_rgb = []
    pcd_semantic = []
    for pcd, (image2, image3), o in zip(pcd_iterator, rgb_iterator, oxt_iterator):
        p = pcd.copy()
        p[:, 3] = 1.0

        imu2world = o.T_w_imu
        velo2world = imu2world @ velo2imu

        world_p = (velo2world @ p.T).T
        world_p = world_p[:, :3] / world_p[:, 3][..., None]

        cam2_coord = (velo2cam2 @ p.T).T
        front_mask = cam2_coord[..., 2] > 0
        cam2_coord = cam2_coord[:, :3] / cam2_coord[:, 2][..., None]

        cam2_pixel_coord = (intr_cam2 @ cam2_coord.T).T
        cam2_mask = (cam2_pixel_coord[..., 0] > 0) * (cam2_pixel_coord[..., 0] < w) * (cam2_pixel_coord[..., 1] > 0) * (cam2_pixel_coord[..., 1] < h) * front_mask
        cam2_pixel_coord = cam2_pixel_coord[cam2_mask].astype(np.int64) # WH

        mask2_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(image_id))
        semantic_map2 = np.load(mask2_path)['arr_0']
        semantic_map2 = cityscapes2concerned(semantic_map2)

        certain_mask2 = getCertainSemanticMask(semantic_map2, cam2_pixel_coord)
        valid_cam2_pixel_coord = cam2_pixel_coord[certain_mask2]
        pcd_xyz.append(world_p[:, :3][cam2_mask][certain_mask2])
        pcd_rgb.append(np.array(image2)[valid_cam2_pixel_coord[..., 1], valid_cam2_pixel_coord[..., 0]])
        pcd_semantic.append(semantic_map2[valid_cam2_pixel_coord[..., 1], [valid_cam2_pixel_coord[..., 0]]][0])

        cam3_coord = (velo2cam3 @ p.T).T
        front_mask = cam3_coord[..., 2] > 0
        cam3_coord = cam3_coord[:, :3] / cam3_coord[:, 2][..., None]

        cam3_pixel_coord = (intr_cam3 @ cam3_coord.T).T
        cam3_mask = (cam3_pixel_coord[..., 0] > 0) * (cam3_pixel_coord[..., 0] < w) * (cam3_pixel_coord[..., 1] > 0) * (cam3_pixel_coord[..., 1] < h) * front_mask
        cam3_pixel_coord = cam3_pixel_coord[cam3_mask].astype(np.int64)

        mask3_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(image_id + n_frames))
        semantic_map3 = np.load(mask3_path)['arr_0']
        semantic_map3 = cityscapes2concerned(semantic_map3)

        certain_mask3 = getCertainSemanticMask(semantic_map3, cam3_pixel_coord)
        valid_cam3_pixel_coord = cam3_pixel_coord[certain_mask3]
        pcd_xyz.append(world_p[:, :3][cam3_mask][certain_mask3])
        pcd_rgb.append(np.array(image3)[valid_cam3_pixel_coord[..., 1], valid_cam3_pixel_coord[..., 0]])
        pcd_semantic.append(semantic_map3[valid_cam3_pixel_coord[..., 1], [valid_cam3_pixel_coord[..., 0]]][0])

        image_id += 1

    pcd_xyz = np.concatenate(pcd_xyz, axis=0)
    pcd_rgb = np.concatenate(pcd_rgb, axis=0).astype(np.float)
    pcd_semantic = np.concatenate(pcd_semantic, axis=0)

    semantic_pcd = SemanticPointCloud(points=pcd_xyz,
                                      colors = pcd_rgb,
                                      semantics=pcd_semantic,
                                      semantics_dict=concerned_classes_ind_map)
    semantic_pcd = semantic_pcd.voxel_down_sample(downsample_lidar_voxel_size)

    output_xyz = semantic_pcd.points.cpu().numpy()
    output_rgb = semantic_pcd.colors.to(torch.uint8).cpu().numpy()
    output_semantic = semantic_pcd.semantics.to(torch.uint8).cpu().numpy()
    storePly(lidar_ply_path, output_xyz, output_rgb)
    storeSemanticPly(semantic_color_ply_path, output_xyz, output_semantic)
    torch.save(semantic_pcd.semantics.to(torch.uint8).cpu(), ply_semantic_index_path)

    return


def getKittiSemanticPcd(
        kitti_data: pykitti.raw,
        colmap_path: str,
        images: str,
        downsample_lidar_voxel_size: float = 0.1,
):
    kitti2colmap = getKitti2Colmap(kitti_data, colmap_path, images)

    lidar_ply_path = os.path.join(colmap_path, "lidar_points3D_{}.ply".format(downsample_lidar_voxel_size))
    semantic_color_ply_path = os.path.join(colmap_path, "lidar_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    ply_semantic_index_path = os.path.join(colmap_path, "lidar_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    if (
            not os.path.exists(lidar_ply_path)
            or not os.path.exists(semantic_color_ply_path)
            or not os.path.exists(ply_semantic_index_path)
    ):
        generateKittiSemanticLidarPcd(kitti_data, colmap_path, downsample_lidar_voxel_size,
                                      lidar_ply_path, semantic_color_ply_path, ply_semantic_index_path)

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
        sfm_xyz, sfm_rgb, sfm_semantic = getKittiColmapSemanticPcd(kitti_data, colmap_path, images)
        sfm_pcd = SemanticPointCloud(points=sfm_xyz, colors=sfm_rgb, semantics=sfm_semantic)

        final_xyz = torch.cat([lidar_pcd.points, sfm_pcd.points], dim=0)
        final_rgb = torch.cat([lidar_pcd.colors, sfm_pcd.colors], dim=0) * 255.0
        final_semantic = torch.cat([lidar_pcd.semantics, sfm_pcd.semantics], dim=0)

        final_xyz_homo = torch.cat([final_xyz, torch.ones(final_xyz.shape[0], 1).to(final_xyz)], dim=-1).double()
        final_xyz_homo = (torch.from_numpy(kitti2colmap).cuda() @ final_xyz_homo.T).T
        final_xyz = final_xyz_homo[..., :3] / final_xyz_homo[..., 3][..., None]

        storePly(final_ply_path, final_xyz.cpu().numpy(), final_rgb.cpu().numpy())
        storeSemanticPly(final_semantic_color_ply_path, final_xyz.cpu().numpy(), final_semantic.cpu().numpy())
        torch.save(final_semantic, final_ply_semantic_index_path)

    final_pcd = fetchPly(final_ply_path)
    final_pcd_semantic = torch.load(final_ply_semantic_index_path)
    final_pcd.load_semantics(final_pcd_semantic)

    return lidar_pcd, final_pcd, lidar_ply_path, final_ply_path


def getKittiColmapSemanticPcd(
        kitti_data: pykitti.raw,
        colmap_path: str,
        images: str,
):
    w, h = kitti_data.get_cam2(0).size
    n_frames = len(kitti_data)

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

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

    kitti2colmap = getKitti2Colmap(kitti_data, colmap_path, images)

    colmap2kitti = np.linalg.inv(kitti2colmap)

    sfm_pcd = fetchPly(colmap_ply_path)
    sfm_pcd_rgb = sfm_pcd.colors.cpu().numpy()
    sfm_pcd_xyz = sfm_pcd.points.cpu().numpy()
    sfm_pcd_xyz_homo = np.concatenate([sfm_pcd_xyz, np.ones(sfm_pcd_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    sfm_pcd_xyz_homo = (colmap2kitti @ sfm_pcd_xyz_homo.T).T

    imu2velo = kitti_data.calib.T_velo_imu
    velo2imu = np.linalg.inv(kitti_data.calib.T_velo_imu)

    velo2cam2 = kitti_data.calib.T_cam2_velo
    velo2cam3 = kitti_data.calib.T_cam3_velo
    intr_cam2 = kitti_data.calib.K_cam2
    intr_cam3 = kitti_data.calib.K_cam3

    sum_semantic = np.zeros((sfm_pcd_xyz.shape[0], len(concerned_classes_list)))
    counts = np.zeros(sfm_pcd_xyz.shape[0])[..., None]

    image_id = 0
    for o in kitti_data.oxts:
        imu2world = o.T_w_imu
        world2imu = np.linalg.inv(imu2world)
        world2velo = imu2velo @ world2imu

        current_velo_pcd_homo = (world2velo @ sfm_pcd_xyz_homo.T).T

        cam2_coord = (velo2cam2 @ current_velo_pcd_homo.T).T
        front_mask = cam2_coord[..., 2] > 0
        cam2_coord = cam2_coord[:, :3] / cam2_coord[:, 2][..., None]

        cam2_pixel_coord = (intr_cam2 @ cam2_coord.T).T
        cam2_mask = (cam2_pixel_coord[..., 0] > 0) * (cam2_pixel_coord[..., 0] < w) * (cam2_pixel_coord[..., 1] > 0) * (cam2_pixel_coord[..., 1] < h) * front_mask
        cam2_pixel_coord = cam2_pixel_coord[cam2_mask].astype(np.int64) # WH

        mask2_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(image_id))
        semantic_map2 = np.load(mask2_path)['arr_0']
        semantic_map2 = cityscapes2concerned(semantic_map2)

        certain_mask2 = getCertainSemanticMask(semantic_map2, cam2_pixel_coord)
        valid_cam2_pixel_coord = cam2_pixel_coord[certain_mask2]

        select_semantic = semantic_map2[valid_cam2_pixel_coord[..., 1], [valid_cam2_pixel_coord[..., 0]]]

        temp = cam2_mask.copy()
        temp[temp == True] = certain_mask2

        final_cam2_mask = temp.copy()

        sum_semantic[final_cam2_mask, select_semantic] += 1
        counts[final_cam2_mask] += 1

        # cam3
        cam3_coord = (velo2cam3 @ current_velo_pcd_homo.T).T
        front_mask = cam3_coord[..., 2] > 0
        cam3_coord = cam3_coord[:, :3] / cam3_coord[:, 2][..., None]

        cam3_pixel_coord = (intr_cam3 @ cam3_coord.T).T
        cam3_mask = (cam3_pixel_coord[..., 0] > 0) * (cam3_pixel_coord[..., 0] < w) * (cam3_pixel_coord[..., 1] > 0) * (cam3_pixel_coord[..., 1] < h) * front_mask
        cam3_pixel_coord = cam3_pixel_coord[cam3_mask].astype(np.int64) # WH

        mask3_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(image_id + n_frames))
        semantic_map3 = np.load(mask3_path)['arr_0']
        semantic_map3 = cityscapes2concerned(semantic_map3)

        certain_mask3 = getCertainSemanticMask(semantic_map3, cam3_pixel_coord)
        valid_cam3_pixel_coord = cam3_pixel_coord[certain_mask3]

        select_semantic = semantic_map3[valid_cam3_pixel_coord[..., 1], [valid_cam3_pixel_coord[..., 0]]]

        temp = cam3_mask.copy()
        temp[temp == True] = certain_mask3

        final_cam3_mask = temp.copy()

        sum_semantic[final_cam3_mask, select_semantic] += 1
        counts[final_cam3_mask] += 1

        image_id += 1

    valid_mask = (counts > 0).squeeze(-1)

    sfm_pcd_xyz = sfm_pcd_xyz_homo[..., :3] / sfm_pcd_xyz_homo[..., 3][..., None]
    sfm_pcd_xyz = sfm_pcd_xyz[valid_mask]
    sfm_pcd_rgb = sfm_pcd_rgb[valid_mask]
    semantic_tag = np.argmax(sum_semantic[valid_mask], axis=1)

    assert semantic_tag.shape[0] == sfm_pcd_xyz.shape[0]
    return sfm_pcd_xyz, sfm_pcd_rgb, semantic_tag




def readKittiInfo(
        kitti_path: str,
        images: str,
        colmap_path: str,
        eval: bool = False,
        llffhold: int = 8,
):
    colmap_path_parts = os.path.normpath(colmap_path).split(os.sep)
    drive = re.findall(r'\d{4}_sync', str(colmap_path_parts))
    drive = drive[0][:-5]

    date = re.findall(r'\d{4}_\d{2}_\d{2}', str(colmap_path_parts))
    assert date[0] == date[1]
    date = date[0]

    cam2_dir = os.path.join(kitti_path, date, "{}_drive_{}_sync".format(date, drive), "image_02", "data")
    cam3_dir = os.path.join(kitti_path, date, "{}_drive_{}_sync".format(date, drive), "image_03", "data")
    num_cam2 = count_png_files(cam2_dir)
    num_cam3 = count_png_files(cam3_dir)
    assert num_cam2 == num_cam3

    kitti_data = pykitti.raw(kitti_path, date, drive, frames=range(0, num_cam2, 1))

    lidar_pcd, final_pcd, lidar_ply_path, final_ply_path = getKittiSemanticPcd(kitti_data, colmap_path, images)

    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

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
    camera_frame_dict['front_end'] = len(kitti_data)

    return scene_info, camera_frame_dict


