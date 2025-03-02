
CUDA_VISIBLE_DEVICES=$3 python3 eval_lpips.py --eval_path $1 --gt_path $2

python3 -m pytorch_fid $1 $2  --device cuda:$3
