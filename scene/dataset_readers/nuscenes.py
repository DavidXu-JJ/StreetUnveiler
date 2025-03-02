
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
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud as NuScenesLidarPointCloud
from nuscenes.utils.geometry_utils import view_points


from utils.pcd_utils import SemanticPointCloud
from utils.semantic_utils import cityscapes2concerned, concerned_classes_list, concerned_classes_ind_map
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers.basic_utils import CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly, storeSemanticPly, to_homo_np
from scene.dataset_readers.colmap import readColmapCameras
from scene.dataset_readers.projection_utils import getCullMaskPointCloudInFrame, getCertainSemanticMask

# Get the frame id of the same time(contains all cameras)
def getNuScenesFrameTokenList(
        nusc: NuScenes,
        scene_id: int,
):
    scene = nusc.scene[scene_id]

    frame_token_list = []

    current_token = scene['first_sample_token']
    last_token = scene['last_sample_token']

    while current_token != last_token:
        frame_token_list.append(current_token)
        current_token_sample = nusc.get('sample', current_token)
        current_token = current_token_sample['next']

    frame_token_list.append(current_token)

    return frame_token_list

def mapNuScenesPcdToImage(
        nusc: NuScenes,
        pc: NuScenesLidarPointCloud,
        camera_token: str,
        pointsensor,
        min_dist: float = 1.0,
) -> Tuple:

    cam = nusc.explorer.nusc.get('sample_data', camera_token)

    with Image.open(os.path.join(nusc.explorer.nusc.dataroot, cam['filename'])) as img:
        im = np.array(img)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.explorer.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.explorer.nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    global_pc = pc.points[:3, :].copy()

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.explorer.nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.explorer.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask].astype(int)
    coloring = im[points[1, ...], points[0, ...], :]

    return global_pc[:3, mask].transpose(1, 0), coloring, points[:2, :].transpose(1,0)

def generateNuScenesSemanticLidarPcd(
        nusc: NuScenes,
        scene_id: int,
        colmap_path: str,
        downsample_lidar_voxel_size: float = 0.1,
):
    def idx_to_frame_str(frame_index):
        return f'{frame_index:08d}'

    def idx_to_mask_filename(frame_index, compress=True):
        ext = 'npz' if compress else 'npy'
        return f'{idx_to_frame_str(frame_index)}.{ext}'

    frame_token_list = getNuScenesFrameTokenList(nusc, scene_id)
    consider_camera = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    lidar_ply_path = os.path.join(colmap_path, "lidar_points3D_{}.ply".format(downsample_lidar_voxel_size))
    semantic_color_ply_path = os.path.join(colmap_path, "lidar_semantic_points3D_{}.ply".format(downsample_lidar_voxel_size))
    ply_semantic_index_path = os.path.join(colmap_path, "lidar_semantic_index_{}.pt".format(downsample_lidar_voxel_size))

    frame_num = len(frame_token_list)
    pcd_list = []
    rgb_list = []
    semantic_list = []
    for frame_idx, frame_token in enumerate(frame_token_list):
        sample = nusc.get('sample', frame_token)
        pointsensor_token = sample['data']['LIDAR_TOP']
        pointsensor = nusc.explorer.nusc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(nusc.explorer.nusc.dataroot, pointsensor['filename'])

        for camera_idx, camera in enumerate(consider_camera):
            pc = NuScenesLidarPointCloud.from_file(pcl_path)
            camera_token = nusc.get('sample', frame_token)['data'][camera]
            xyz, rgb, projectionWH = mapNuScenesPcdToImage(nusc, pc, camera_token, pointsensor)

            mask_idx = camera_idx * frame_num + frame_idx
            mask_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(mask_idx))
            semantic_map = np.load(mask_path)['arr_0']
            semantic_map = cityscapes2concerned(semantic_map)

            certain_semantic_mask = getCertainSemanticMask(semantic_map, projectionWH)

            xyz = xyz[certain_semantic_mask]
            rgb = rgb[certain_semantic_mask]
            semantic = semantic_map[projectionWH[..., 1], projectionWH[..., 0]][certain_semantic_mask]

            pcd_list.append(xyz)
            rgb_list.append(rgb)
            semantic_list.append(semantic)

    xyz_np = np.concatenate(pcd_list, axis=0)
    rgb_np = np.concatenate(rgb_list, axis=0).astype(float)
    semantic_np = np.concatenate(semantic_list, axis=0)

    semantic_pcd = SemanticPointCloud(points=xyz_np,
                                      colors=rgb_np,
                                      semantics=semantic_np,
                                      semantics_dict=concerned_classes_ind_map)
    # semantic_pcd = semantic_pcd.voxel_down_sample(downsample_lidar_voxel_size)

    output_xyz = semantic_pcd.points.cpu().numpy()
    output_rgb = semantic_pcd.colors.to(torch.uint8).cpu().numpy()
    output_semantic = semantic_pcd.semantics.to(torch.uint8).cpu().numpy()
    storePly(lidar_ply_path, output_xyz, output_rgb)
    storeSemanticPly(semantic_color_ply_path, output_xyz, output_semantic)
    torch.save(semantic_pcd.semantics.to(torch.uint8).cpu(), ply_semantic_index_path)

    return

def getNuScenesColmapSemanticPcd(
        nusc: NuScenes,
        scene_id: int,
        colmap_path: str,
        images: str,
):
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

    NuScenes2Colmap = getNuScenes2Colmap(nusc, scene_id, colmap_path, images)

    Colmap2NuScenes = np.linalg.inv(NuScenes2Colmap)

    sfm_pcd = fetchPly(colmap_ply_path)
    sfm_pcd_xyz = sfm_pcd.points.cpu().numpy()
    sfm_pcd_xyz_homo = np.concatenate([sfm_pcd_xyz, np.ones(sfm_pcd_xyz.shape[:-1])[:, np.newaxis]], axis=-1)
    sfm_pcd_xyz_homo = (Colmap2NuScenes @ sfm_pcd_xyz_homo.T).T
    sfm_pcd_xyz = sfm_pcd_xyz_homo[..., :3] / sfm_pcd_xyz_homo[..., 3][..., None]

    sfm_pcd_rgb = sfm_pcd.colors.cpu().numpy()

    frame_token_list = getNuScenesFrameTokenList(nusc, scene_id)
    consider_camera = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    frame_num = len(frame_token_list)

    sum_semantic = np.zeros((sfm_pcd_xyz.shape[0], len(concerned_classes_list)))
    counts = np.zeros(sfm_pcd_xyz.shape[0])[..., None]
    for frame_idx, frame_token in enumerate(frame_token_list):
        for camera_idx, camera in enumerate(consider_camera):
            camera_token = nusc.get('sample', frame_token)['data'][camera]
            cam = nusc.explorer.nusc.get('sample_data', camera_token)
            pc = NuScenesLidarPointCloud(np.concatenate([sfm_pcd_xyz, np.ones(sfm_pcd_xyz.shape[:-1])[:, np.newaxis]], axis=-1).T)

            # First step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.explorer.nusc.get('ego_pose', cam['ego_pose_token'])
            pc.translate(-np.array(poserecord['translation']))
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

            # Second step: transform from ego into the camera.
            cs_record = nusc.explorer.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            depths = pc.points[2, :]

            points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

            mask_idx = camera_idx * frame_num + frame_idx
            mask_path = os.path.join(colmap_path, "input_masks", idx_to_mask_filename(mask_idx))
            semantic_map = np.load(mask_path)['arr_0']
            semantic_map = cityscapes2concerned(semantic_map)

            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < semantic_map.shape[1] - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < semantic_map.shape[0] - 1)

            points = points[:, mask].astype(int)
            projectionWH = points[:2, :].transpose(1, 0)

            select_semantic = semantic_map[projectionWH[..., 1], projectionWH[..., 0]]

            sum_semantic[mask, select_semantic] += 1
            counts[mask] += 1

    valid_mask = (counts > 0).squeeze(-1)

    sfm_pcd_xyz = sfm_pcd_xyz[valid_mask]
    sfm_pcd_rgb = sfm_pcd_rgb[valid_mask]
    semantic_tag = np.argmax(sum_semantic[valid_mask], axis=1)

    assert semantic_tag.shape[0] == sfm_pcd_xyz.shape[0]
    return sfm_pcd_xyz, sfm_pcd_rgb, semantic_tag


def getNuScenesCameraPosition(
        nusc: NuScenes,
        scene_id: int,
):
    frame_token_list = getNuScenesFrameTokenList(nusc, scene_id)

    positions = []
    consider_camera = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    for camera_idx, camera in enumerate(consider_camera):
        for frame_idx, frame_token in enumerate(frame_token_list):

            camera_token = nusc.get('sample', frame_token)['data'][camera]
            cam = nusc.explorer.nusc.get('sample_data', camera_token)

            poserecord = nusc.explorer.nusc.get('ego_pose', cam['ego_pose_token'])
            transform1 = np.eye(4)
            transform1[:3, 3] -= np.array(poserecord['translation'])

            transform2 = np.eye(4)
            transform2[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix.T

            cs_record = nusc.explorer.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            transform3 = np.eye(4)
            transform3[:3, 3] -= np.array(cs_record['translation'])

            transform4 = np.eye(4)
            transform4[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix.T

            c2w = (np.linalg.inv(transform4 @ transform3 @ transform2 @ transform1))
            point = c2w[:3, 3]
            positions.append(point.reshape(1, 3))

    positions = np.vstack(positions)

    return positions

def getNuScenes2Colmap(
        nusc: NuScenes,
        scene_id: int,
        colmap_path: str,
        images: str,
):
    nusc_cam_positions = getNuScenesCameraPosition(nusc, scene_id)

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

    rmsd, R_ij, T_i, c = Superpose3D(colmap_translation_list, nusc_cam_positions, None, True, False)

    NuScenes2Colmap = np.eye(4)
    NuScenes2Colmap[:3, :3] = np.array(R_ij) * c
    NuScenes2Colmap[:3, 3] = np.array(T_i)

    return NuScenes2Colmap

def getNuScenesSemanticPcd(
        nusc: NuScenes,
        scene_id: int,
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
        generateNuScenesSemanticLidarPcd(nusc, scene_id, colmap_path, downsample_lidar_voxel_size)

    # NuScenes coords
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
        # NuScenes coords
        sfm_xyz, sfm_rgb, sfm_semantic = getNuScenesColmapSemanticPcd(nusc, scene_id, colmap_path, images)
        sfm_pcd = SemanticPointCloud(points=sfm_xyz, colors=sfm_rgb, semantics=sfm_semantic)

        final_xyz = torch.cat([lidar_pcd.points, sfm_pcd.points], dim=0)
        final_rgb = torch.cat([lidar_pcd.colors, sfm_pcd.colors], dim=0) * 255.0
        final_semantic = torch.cat([lidar_pcd.semantics, sfm_pcd.semantics], dim=0)

        NuScenes2Colmap = getNuScenes2Colmap(nusc, scene_id, colmap_path, images)
        final_xyz_homo = torch.cat([final_xyz, torch.ones(final_xyz.shape[0], 1).to(final_xyz)], dim=-1)
        final_xyz_homo = (torch.from_numpy(NuScenes2Colmap).cuda() @ final_xyz_homo.T).T
        final_xyz = final_xyz_homo[..., :3] / final_xyz_homo[..., 3][..., None]

        storePly(final_ply_path, final_xyz.cpu().numpy(), final_rgb.cpu().numpy())
        storeSemanticPly(final_semantic_color_ply_path, final_xyz.cpu().numpy(), final_semantic.cpu().numpy())
        torch.save(final_semantic, final_ply_semantic_index_path)

    final_pcd = fetchPly(final_ply_path)
    final_pcd_semantic = torch.load(final_ply_semantic_index_path)
    final_pcd.load_semantics(final_pcd_semantic)

    return lidar_pcd, final_pcd, lidar_ply_path, final_ply_path

def readNuScenesInfo(
        nusc_path: str,
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

    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_path, verbose=True)

    scene_name = os.path.basename(colmap_path)

    scene_idx = None
    for idx in range(len(nusc.scene)):
        if nusc.scene[idx]['name'] == scene_name:
            scene_idx = idx
            break

    assert scene_idx is not None, f"Scene {scene_name} not found in NuScenes dataset"

    lidar_pcd, final_pcd, lidar_ply_path, final_ply_path = getNuScenesSemanticPcd(nusc, scene_idx, colmap_path, images)

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
    frame_token_list = getNuScenesFrameTokenList(nusc, scene_idx)
    camera_frame_dict['front_start'] = 0
    camera_frame_dict['front_end'] = len(frame_token_list)

    return scene_info, camera_frame_dict