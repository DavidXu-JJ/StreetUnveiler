#!/bin/bash

base_dir=$1

new_dir=$2

subdirs=("image_02" "image_03")

for dir in "$base_dir"/*/; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != "." ] && [ "$(basename "$dir")" != ".." ]; then
        dir=$(basename "$dir")

        src_dir="$base_dir/$dir"
        dst_dir="$new_dir/$dir/input"

        mkdir -p "$dst_dir"

        count=0

        for subdir in "${subdirs[@]}"; do
            if [ -d "$src_dir/$subdir/data" ] && ls "$src_dir/$subdir/data/"*.png 1> /dev/null 2>&1; then
                for file in "$src_dir/$subdir/data/"*.png; do
                    new_filename=$(printf "%08d.jpg" "$count")
                    cp "$file" "$dst_dir/$new_filename"
                    count=$((count+1))
                done
            fi
        done
    fi
done