#!/bin/bash

base_dir=$1
start=$2
end=$3
cuda_devices=$4
count=0

export CUDA_VISIBLE_DEVICES=$cuda_devices

for dir in $(ls $base_dir | sort)
do
    if [ "$count" -lt "$start" ]
    then
        count=$((count+1))
        continue
    fi

    if [ "$count" -ge "$end" ]
    then
        break
    fi

    src_dir="$base_dir/$dir"

    python3 convert.py -s $src_dir/$subdir
    python3 convert.py -s $src_dir/$subdir --skip_matching

    count=$((count+1))
done