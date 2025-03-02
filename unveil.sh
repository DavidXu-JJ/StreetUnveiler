
# sh unveil.sh model_name front_key_frames gpu_id

CUDA_VISIBLE_DEVICES=$3 python3 inpainting_pipeline/3_reoptimization/1_optimization.py -m $1 --front_key_frames $2

CUDA_VISIBLE_DEVICES=$3 python3 inpainting_pipeline/3_reoptimization/2_visualization.py -m $1