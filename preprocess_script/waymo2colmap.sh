#!/bin/bash

base_dir=$1

new_dir=$2

subdirs=("camera_FRONT" "camera_FRONT_LEFT" "camera_FRONT_RIGHT")

for dir in $(ls $base_dir | sort)
do
    src_dir="$base_dir/$dir/images"
    dst_dir="$new_dir/$dir/input"

    mkdir -p $dst_dir

    count=0

    for subdir in "${subdirs[@]}"
    do
        if [ -d "$src_dir/$subdir" ] && ls $src_dir/$subdir/*.jpg 1> /dev/null 2>&1; then
            for file in $(ls $src_dir/$subdir/*.jpg | sort)
            do
                new_filename=$(printf "%08d.jpg" $count)
                cp $file $dst_dir/$new_filename
                count=$((count+1))
            done
        fi
    done
done