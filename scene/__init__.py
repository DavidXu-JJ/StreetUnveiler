#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

import os
import random
import json
import numpy as np
from collections import defaultdict
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.env_map import SkyModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import fov2focal, getProjectionMatrix
from utils.semantic_utils import concerned_classes_ind_map

class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, sky_model: SkyModel=None, load_iteration=None, shuffle=False, resolution_scales=[1.0], only_pose=False, splatting_ply_path=None):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_type = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            self.scene_type = "colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            self.scene_type = "blender"
        elif os.path.exists(os.path.join(args.source_path, "scenario.pt")):
            print("Found scenario.pt file, assuming Waymo data set!")
            scene_info, camera_frame_dict = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.images, args.colmap_path, args.eval)
            self.scene_type = "waymo"
            self.camera_frame_dict = camera_frame_dict
        elif os.path.exists(os.path.join(args.source_path, "Pandaset")):
            print("Found pandaset!")
            scene_info, camera_frame_dict = sceneLoadTypeCallbacks["Pandaset"](args.source_path, args.images, args.colmap_path, args.eval)
            self.scene_type = "pandaset"
            self.camera_frame_dict = camera_frame_dict
        elif os.path.exists(os.path.join(args.source_path, "raw_data_downloader.sh")):
            print("Found raw_data_downloader.sh file, assuming Kitti data set!")
            scene_info, camera_frame_dict = sceneLoadTypeCallbacks["Kitti"](args.source_path, args.images, args.colmap_path, args.eval)
            self.scene_type = "kitti"
        elif os.path.exists(os.path.join(args.source_path, "v1.0-trainval")):
            print("Found v1.0-trainval folder, assuming nuScenes data set!")
            scene_info, camera_frame_dict = sceneLoadTypeCallbacks["NuScenes"](args.source_path, args.images, args.colmap_path, args.eval)
            self.scene_type = "nuscenes"
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            with open(scene_info.reference_ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "reference.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.train_index_list = [i for i in range(len(scene_info.train_cameras))]
        self.n_images = len(self.train_index_list)

        if shuffle:
            random.shuffle(self.train_index_list)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.reso_scale_list = defaultdict(list)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale], reso_scale_list = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, only_pose)

            self.reso_scale_list[resolution_scale] = reso_scale_list
            if shuffle:
                self.train_cameras[resolution_scale] = [
                    self.train_cameras[resolution_scale][i] for i in self.train_index_list
                ]

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale], _ = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, only_pose)

        if self.loaded_iter:
            # CHECK: the continue need to load the reference point cloud, it should be saved.
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply") if splatting_ply_path is None else splatting_ply_path)
            if sky_model is not None:
                sky_model.load(os.path.join(self.model_path, "checkpoint", "iteration_" + str(self.loaded_iter), "sky_params.pt"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.w2c_list = defaultdict(list)
        self.intr_list = defaultdict(list)
        self.H_list = defaultdict(list)
        self.W_list = defaultdict(list)
        self.full_proj_matrix_list = defaultdict(list)
        for resolution_scale in resolution_scales:
            for frame_id in range(len(scene_info.train_cameras)):
                frame_cam = self.train_cameras[resolution_scale][frame_id]
                fovx = frame_cam.FoVx
                fovy = frame_cam.FoVy

                H = frame_cam.image_height
                W = frame_cam.image_width

                R = frame_cam.R
                T = frame_cam.T

                w2c = torch.eye(4)
                w2c[:3, :3] = torch.from_numpy(np.linalg.inv(R))
                w2c[:3, 3] = torch.from_numpy(T)

                intr = torch.zeros((3, 4))
                intr[0, 0] = fov2focal(fovx, W)
                intr[1, 1] = fov2focal(fovy, H)

                intr[0, 2] = W / 2.
                intr[1, 2] = H / 2.
                intr[2, 2] = 1.0

                prospective2orghogral = getProjectionMatrix(
                    znear = 0.01,
                    zfar = 100.0,
                    fovX = fovx,
                    fovY = fovy,
                    K = intr,
                    img_h = H,
                    img_w = W,
                )

                full_proj_matrix = prospective2orghogral @ w2c

                self.w2c_list[resolution_scale].append(w2c.cuda())
                self.intr_list[resolution_scale].append(intr.cuda())
                self.H_list[resolution_scale].append(H)
                self.W_list[resolution_scale].append(W)
                self.full_proj_matrix_list[resolution_scale].append(full_proj_matrix.cuda())

    def getH(self, idx, scale=1.0):
        return self.H_list[scale][idx]

    def getW(self, idx, scale=1.0):
        return self.W_list[scale][idx]

    def getIntr(self, idx, scale=1.0):
        return self.intr_list[scale][idx]

    def getExtr(self, idx, scale=1.0):
        return torch.inverse(self.w2c_list[scale][idx])

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_semantic_ply(os.path.join(point_cloud_path, "semantic_point_cloud.ply"))
        self.gaussians.save_opacity_ply(os.path.join(point_cloud_path, "opacity_point_cloud.ply"))

    def save_at_inpaint(self, loaded_iter, iteration):
        point_cloud_path = os.path.join(self.model_path, "inpaint_{}".format(loaded_iter), "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_semantic_ply(os.path.join(point_cloud_path, "semantic_point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getNearestTrainCamerasId(self, points, scale=1.0):
        mean_point = points.mean(0)

        min_distance = 1e9
        nearest_camera = None
        for frame_id in range(len(self.train_cameras[scale])):
            position = self.w2c_list[scale][frame_id][:3, 3]
            distance = torch.norm(position - mean_point)
            if distance < min_distance:
                min_distance = distance
                nearest_camera = frame_id

        return nearest_camera

    def getTrainCamerasOrigin(self, frame_id, scale=1.0):
        w2c = self.w2c_list[scale][frame_id]
        return w2c[:3, 3]

    def getTrainCamerasTopAxis(self, frame_id, scale=1.0):
        w2c = self.w2c_list[scale][frame_id]
        return (- w2c[:3, 1])

    def getPcdInTrainFrame(self, pcd, frame_id, scale=1.0):
        lidar_pcd = pcd
        w2c = self.w2c_list[scale][frame_id]
        camera_point = (w2c[:3, :3] @ lidar_pcd.T).T + w2c[:3, 3]
        positive_z_mask = camera_point[..., 2] > 0

        pixel_point = (self.intr_list[scale][frame_id][:3, :3] @ camera_point.T).T
        # [w, h] = [x, y]
        pix = pixel_point[..., :2] / pixel_point[..., 2][..., None]

        h = self.H_list[scale][frame_id]
        w = self.W_list[scale][frame_id]
        mask = (pix[..., 0] < w) * (pix[..., 0] > 0) * (pix[..., 1] < h) * (pix[..., 1] > 0) * positive_z_mask

        ret_pcd = lidar_pcd[mask]
        return ret_pcd, mask

    def getPcdPixelCoordsInTrainFrame(self, pcd, frame_id, max_z_val=10.0, scale=1.0):
        lidar_pcd = pcd
        w2c = self.w2c_list[scale][frame_id]
        camera_point = (w2c[:3, :3] @ lidar_pcd.T).T + w2c[:3, 3]
        positive_z_mask = camera_point[..., 2] > 0
        if max_z_val is None:
            small_z_mask = torch.ones_like(camera_point[..., 2], dtype=torch.bool)
        else:
            small_z_mask = camera_point[..., 2] < max_z_val

        pixel_point = (self.intr_list[scale][frame_id][:3, :3] @ camera_point.T).T
        # [w, h] = [x, y]
        pix = pixel_point[..., :2] / pixel_point[..., 2][..., None]

        h = self.H_list[scale][frame_id]
        w = self.W_list[scale][frame_id]
        in_frame_mask = (pix[..., 0] < w) * (pix[..., 0] > 0) * (pix[..., 1] < h) * (pix[..., 1] > 0) * positive_z_mask * small_z_mask

        valid_pix = pix[in_frame_mask].to(torch.int64)
        if self.reso_scale_list[scale][frame_id] is not None:
            valid_pix = (valid_pix / self.reso_scale_list[scale][frame_id]).to(torch.int64)

        # [ num_valid_points, 2(which represent xy/wh) ]
        return valid_pix, in_frame_mask

    def getPcdPixelCoordsInTrainFrameWithDepth(self, pcd, frame_id, max_z_val=10.0, scale=1.0):
        lidar_pcd = pcd
        w2c = self.w2c_list[scale][frame_id]
        camera_point = (w2c[:3, :3] @ lidar_pcd.T).T + w2c[:3, 3]
        positive_z_mask = camera_point[..., 2] > 0
        if max_z_val is None:
            small_z_mask = torch.ones_like(camera_point[..., 2], dtype=torch.bool)
        else:
            small_z_mask = camera_point[..., 2] < max_z_val

        pixel_point = (self.intr_list[scale][frame_id][:3, :3] @ camera_point.T).T
        # [w, h] = [x, y]
        pix = pixel_point[..., :2] / pixel_point[..., 2][..., None]

        h = self.H_list[scale][frame_id]
        w = self.W_list[scale][frame_id]
        in_frame_mask = (pix[..., 0] < w) * (pix[..., 0] > 0) * (pix[..., 1] < h) * (pix[..., 1] > 0) * positive_z_mask * small_z_mask

        valid_pix = pix[in_frame_mask].to(torch.int64)
        if self.reso_scale_list[scale][frame_id] is not None:
            valid_pix = (valid_pix / self.reso_scale_list[scale][frame_id]).to(torch.int64)

        valid_pcd_depth = camera_point[in_frame_mask][..., 2]

        # [ num_valid_points, 2(xy/wh) ]
        return valid_pix, valid_pcd_depth

    """ get the project semantic mask in predicted semantic map of splatting with certain semantic tag"""
    def getSemanticMaskOfSplatting(self, semantic_remain_bit, scale=1.0):
        with torch.no_grad():
            final_mask = torch.zeros((self.gaussians.get_xyz.shape[0],), dtype=torch.bool)
            index_tensor = torch.arange(self.gaussians.get_xyz.shape[0])

            train_cams = self.getTrainCameras(scale)
            for frame_id in range(self.n_images):
                current_semantic_map = train_cams[frame_id].semantic_map

                # [in_frame_pix_num], [all_splatting_num]
                valid_pix, in_frame_mask = self.getPcdPixelCoordsInTrainFrame(self.gaussians.get_xyz, frame_id, None, scale)

                # [in_frame_pix_num]
                valid_pix_semantic = current_semantic_map[valid_pix[..., 1], valid_pix[..., 0]]

                # [in_frame_pix_num]
                if_in_semantic_region = ( (1 << valid_pix_semantic.to(torch.int32)) & semantic_remain_bit ) > 0

                # [in_frame_pix_num]
                in_frame_index = index_tensor[in_frame_mask]
                # [in_frame_and_semantic_num]
                in_frame_and_semantic_index = in_frame_index[if_in_semantic_region]

                final_mask[in_frame_and_semantic_index] = True

        return final_mask


