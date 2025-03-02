
# Waymo Open Dataset Data Preparation

The waymo preparation is from and based on the preprocess implementation from awesome project [neuralsim/StreetSurf](https://github.com/PJLab-ADG/neuralsim). Here are detailed steps to download and preprocess the Waymo Open Dataset.

Warning: The segmentation at Step 4 requires a rigid CUDA 11.1. We provide a docker image below for your convenience.

## Step 0: Install environment for downloading and preprocessing only.

```
conda create -n waymo_preprocess python=3.10
conda activate waymo_preprocess

# Suppose CUDA version is 12.1, please change your code properly according to your CUDA version.
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install torch-scatter (pick one from following 2 options)
# If you are using Linux
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp310-cp310-linux_x86_64.whl
# If you are not using Linux, you may try
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+12.1.html

pip install opencv-python-headless kornia imagesize omegaconf addict \
  imageio imageio-ffmpeg scikit-image scikit-learn pyyaml pynvml psutil \
  seaborn==0.12.0 trimesh plyfile ninja icecream tqdm plyfile tensorboard \
  torchmetrics
 
pip install tensorflow_gpu==2.11.0 waymo-open-dataset-tf-2-11-0
```

## Step 1: Install Google Cloud CLI for downloading Waymo Open Dataset.

- Please follow the official 7-step instruction to install Google Cloud CLI: https://cloud.google.com/sdk/docs/install

```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz

tar -xf google-cloud-cli-linux-x86_64.tar.gz

./google-cloud-sdk/install.sh

./google-cloud-sdk/bin/gcloud init
```

Don't forget to add `bin` folder to your PATH environment variable.

- Fill out the Waymo Terms of Use agreement. Checkout the official link at https://waymo.com/open/download

- `gcloud auth login` to login your Google account to make yourself authorized to download the dataset.

## Step 2: Download and Preprocess Waymo Open Dataset.

```
cd ./3rd_party/neuralsim/dataio/autonomous_driving/waymo
bash download_waymo.sh waymo_static_32.lst ../../../../../data/waymo/training
python preprocess.py --root=../../../../../data/waymo/training --out_root=../../../../../data/waymo/processed -j4 --seq_list=waymo_static_32.lst
```

While this paper StreetUnveiler mainly targets on the static scenes, you can also download the other kinds of scenes by editing the `waymo_static_32.lst`.

## Step 3: Run COLMAP for SfM point cloud

Please ensure COLMAP executable is installed properly. You can refer to the [official website](https://colmap.github.io/install.html) for installation.

Note: It's more recommended to use docker for the pre-installed COLMAP environment. You can refer to the [official docker image](https://hub.docker.com/r/colmap/colmap) for more details.

```
# copy images
cd ./preprocess_script
bash waymo2colmap.sh ../data/waymo/processed ../data/waymo/colmap

# run colmap for all scenes
# (change the argument properly)
# Explain: run_colmap.sh [colmap_dir] [start_scene_index] [end_scene_index] [used_gpu_index]
sh run_colmap.sh ../data/waymo/colmap 0 1 0
```

For now, the `./data/waymo` data directory should look like:

```
data
|-- waymo
    |-- colmap
        |-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
            |-- distorted
            |-- images
            |-- input
            |-- run-colmap-geometric.sh
            |-- run-colmap-photometric.sh
            |-- sparse
            `-- stereo
    |-- processed
        |-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
            |-- images
            |-- lidars
            `-- scenario.pt
```

Note: It's possible that the number of images in folder `images` is not the same as in `input`, or getting some weird distorted images in `images` folder. This may be solved by re-running the `run_colmap.sh` script.

(We prepare the results of some scenes after running COLMAP. If you want to skip this step, please run the following command.)

```bash
cd data/waymo
wget https://huggingface.co/jingwei-xu-00/pretrained_backup_for_streetunveiler/resolve/main/waymo_colmap_result.zip
unzip waymo_colmap_result.zip
mv waymo_colmap_result colmap
```

## Step 4: Run image semantic segmentation

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
    --data_root ../../../../../data/waymo/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname images \
    --mask_dirname images_masks
    
python3 extract_masks_after_colmap.py \
    --data_root ../../../../../data/waymo/colmap \
    --segformer_path ../../../../SegFormer \
    --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
    --rgb_dirname input \
    --mask_dirname input_masks

```

Finally, the `./data/waymo` data directory should look like:

```
data
|-- waymo
    |-- colmap
    |   |-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
    |       |-- distorted
    |       |-- images
    |       |-- images_masks
    |       |-- input
    |       |-- input_masks
    |       |-- run-colmap-geometric.sh
    |       |-- run-colmap-photometric.sh
    |       |-- sparse
    |       `-- stereo
    `-- processed
        |-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
            |-- images
            |-- lidars
            `-- scenario.pt
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

```bibtex
@article{guo2023streetsurf,
  title = {StreetSurf: Extending Multi-view Implicit Surface Reconstruction to Street Views},
  author = {Guo, Jianfei and Deng, Nianchen and Li, Xinyang and Bai, Yeqi and Shi, Botian and Wang, Chiyu and Ding, Chenjing and Wang, Dongliang and Li, Yikang},
  journal = {arXiv preprint arXiv:2306.04988},
  year = {2023}
}
```