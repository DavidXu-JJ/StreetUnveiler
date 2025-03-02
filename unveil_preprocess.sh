
# sh unveil_preprocess.sh model_name gpu_id

CUDA_VISIBLE_DEVICES=$2 python3 inpainting_pipeline/1_selection/1_instance_visualization.py -m $1 --mask_vehicle # --mask_vehicle option can be modified to change the semantic of the removed objects

CUDA_VISIBLE_DEVICES=$2 python3 inpainting_pipeline/2_condition_preparation/1_select_instance.py -m $1 --all

CUDA_VISIBLE_DEVICES=$2 python3 inpainting_pipeline/2_condition_preparation/2_generate_inpainted_mask.py -m $1