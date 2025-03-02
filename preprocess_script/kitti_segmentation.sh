#!/bin/bash

cd ../3rd_party/neuralsim/dataio/autonomous_driving/waymo

DATA_ROOT="../../../../../data/kitti/colmap"

for subdir in "$DATA_ROOT"/*; do
    if [ -d "$subdir" ]; then  # 确保是文件夹
        echo "Processing $subdir"

        python3 extract_masks_after_colmap.py \
            --data_root "$subdir" \
            --segformer_path ../../../../SegFormer \
            --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
            --rgb_dirname images \
            --mask_dirname images_masks

        python3 extract_masks_after_colmap.py \
            --data_root "$subdir" \
            --segformer_path ../../../../SegFormer \
            --checkpoint ../../../segformer.b5.1024x1024.city.160k.pth \
            --rgb_dirname input \
            --mask_dirname input_masks
    fi
done