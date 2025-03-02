
# nuScenes Data Preparation

## Step 1: Download the Dataset

Please download the dataset from the official website [nuScenes](https://www.nuscenes.org/nuscenes). 

Placing it under `./data/nuscenes/raw` directory.

## Step 2: Run COLMAP for SfM point cloud

Please ensure COLMAP executable is installed properly. You can refer to the [official website](https://colmap.github.io/install.html) for installation.

Note: It's more recommended to use docker for the pre-installed COLMAP environment. You can refer to the [official docker image](https://hub.docker.com/r/colmap/colmap) for more details.

```
# copy images
cd ./preprocess_script
python3 nuscenes_preprocess.py

# run colmap for all scenes
# (change the argument properly)
# Explain: run_colmap.sh [colmap_dir] [start_scene_index] [end_scene_index] [used_gpu_index]
sh run_colmap.sh ../data/nuscenes/colmap 0 1 0
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

```
cd ./3rd_party/neuralsim/dataio/autonomous_driving/waymo

python3 extract_masks_after_colmap.py \
    --data_root ../../../../../data/nuscenes/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname images \
    --mask_dirname images_masks
    
python3 extract_masks_after_colmap.py \
    --data_root ../../../../../data/nuscenes/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname input \
    --mask_dirname input_masks
```

Finally, the `./data/nuscenes` data directory should look like:

```
data
|-- nuscenes
    |-- colmap
    |   |-- scene-0001
    |   |   |-- distorted
    |   .   |-- images
    |   .   |-- images_masks
    |   .   |-- input
    |       |-- input_masks
    |       |-- run-colmap-geometric.sh
    |       |-- run-colmap-photometric.sh
    |       |-- sparse
    |       `-- stereo
    `-- raw
        |-- maps
        |-- samples
        |-- sweeps
        `-- v1.0-trainval
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
