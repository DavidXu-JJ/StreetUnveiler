
import os
import shutil
from nuscenes.nuscenes import NuScenes

data_root = '../data/nuscenes/raw'
output_root = '../data/nuscenes/colmap'

def get_scene_token_list(nusc, scene_idx):
    scene = nusc.scene[scene_idx]

    frame_token_list = []

    current_token = scene['first_sample_token']
    last_token = scene['last_sample_token']

    while current_token != last_token:
        frame_token_list.append(current_token)
        current_token_sample = nusc.get('sample', current_token)
        current_token = current_token_sample['next']

    frame_token_list.append(current_token)

    return frame_token_list


def get_scene_camera(nusc, scene_idx, camera_name_list):

    frame_token_list = get_scene_token_list(nusc, scene_idx)

    scene_name = nusc.scene[scene_idx]['name']

    count = 0

    for camera_name in camera_name_list:
        scene_img_token = []

        for frame_token in frame_token_list:
            frame = nusc.get('sample', frame_token)
            scene_img_token.append(frame['data'][camera_name])

        scene_img_path = []
        for img_token in scene_img_token:
            img = nusc.get('sample_data', img_token)
            scene_img_path.append(os.path.join(data_root, img['filename']))

        if not os.path.exists(output_root):
            os.mkdir(output_root)

        scene_dir = os.path.join(output_root, scene_name, "input")

        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)

        for img_path in scene_img_path:
            shutil.copy(img_path, os.path.join(scene_dir, "{:08d}.jpg".format(count)))
            count += 1


nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)

scene_num = len(nusc.scene)

for i in range(scene_num):
    get_scene_camera(nusc, i, ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'])