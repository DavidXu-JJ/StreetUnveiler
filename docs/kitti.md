
# Kitti Data Preparation

## Step 1: Download the Dataset

```bash
cd data/kitti/raw
./raw_dara_downloader.sh
```

## Step 2: Run COLMAP for SfM point cloud

Please ensure COLMAP executable is installed properly. You can refer to the [official website](https://colmap.github.io/install.html) for installation.

Note: It's more recommended to use docker for the pre-installed COLMAP environment. You can refer to the [official docker image](https://hub.docker.com/r/colmap/colmap) for more details.

```bash
# copy images
cd ./preprocess_script
bash kitti2colmap.sh ../data/kitti/raw/2011_09_26 ../data/kitti/colmap/2011_09_26
bash kitti2colmap.sh ../data/kitti/raw/2011_09_28 ../data/kitti/colmap/2011_09_28
bash kitti2colmap.sh ../data/kitti/raw/2011_09_29 ../data/kitti/colmap/2011_09_29
bash kitti2colmap.sh ../data/kitti/raw/2011_09_30 ../data/kitti/colmap/2011_09_30
bash kitti2colmap.sh ../data/kitti/raw/2011_10_03 ../data/kitti/colmap/2011_10_03

# run colmap for all scenes
# (change the argument properly)
# Explain: run_colmap.sh [colmap_dir] [start_scene_index] [end_scene_index] [used_gpu_index]
sh run_colmap.sh ../data/kitti/colmap/2011_09_26 0 1 0
sh run_colmap.sh ../data/kitti/colmap/2011_09_28 0 1 0
sh run_colmap.sh ../data/kitti/colmap/2011_09_29 0 1 0
sh run_colmap.sh ../data/kitti/colmap/2011_09_30 0 1 0
sh run_colmap.sh ../data/kitti/colmap/2011_10_03 0 1 0
```

Note: It's possible that the number of images in folder `images` is not the same as in `input`, or getting some weird distorted images in `images` folder. This may be solved by re-running the `run_colmap.sh` script.

## Step 3: Run image semantic segmentation

### Environment

The image semantic segmentation is based on [Segformer](https://github.com/NVlabs/SegFormer). You may follow [neuralsim](https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/README.md#-setup-a-seperate-conda-env-for-segformer) to setup the environment.

Or you may try the docker image provided at [dockerhub](https://hub.docker.com/repository/docker/davidxujw/segformer/general). (`conda activate segformer` to use)

### Pretrained Model

After you set up the environment, please download the pretrained model from [Segformer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#evaluation). Or download from [this backup link](https://drive.google.com/drive/folders/1-FLXx-gNkG-F__F5q64hvXC-yfoC4Wpu?usp=sharing) and place it under `3rd_party/neuralsim`.

Or you may run the following command to download the model:

```bash
cd 3rd_party/neuralsim
wget https://huggingface.co/jingwei-xu-00/pretrained_backup_for_streetunveiler/resolve/main/Segformer/segformer.b5.1024x1024.city.160k.pth
```

Then run the following command to predict the masks:

```bash
cd preprocess_script
bash kitti_segmentation.sh
```

Finally, the `./data/kitti` data directory should look like:

```
data
|--kitti
    |-- colmap
    |   `-- 2011_09_26
    |       |-- 2011_09_26_drive_0001_sync
    |           |-- distorted
    |           |-- images
    |           |-- images_masks
    |           |-- input
    |           |-- input_masks
    |           |-- run-colmap-geometric.sh
    |           |-- run-colmap-photometric.sh
    |           |-- sparse
    |           `-- stereo
    `-- raw
        |-- 2011_09_26
        |   |-- 2011_09_26_drive_0001_sync
        |   |   |-- image_00
        |   |   |-- image_01
        |   |   |-- image_02
        |   |   |-- image_03
        |   |   |-- oxts
        |   |   `-- velodyne_points
        |   |-- calib_cam_to_cam.txt
        |   |-- calib_imu_to_velo.txt
        |   `-- calib_velo_to_cam.txt
        |
        .
        .
        .
        `-- raw_data_downloader.sh

```

# Bibtex

```bibtex
@inproceedings{xu2025streetunveiler,
  author       = {Jingwei Xu and Yikai Wang and Yiqun Zhao and Yanwei Fu and Shenghua Gao},
  title        = {3D StreetUnveiler with Semantic-aware 2DGS - a simple baseline},
  booktitle    = {The International Conference on Learning Representations (ICLR)},
  year         = {2025},
}
```
