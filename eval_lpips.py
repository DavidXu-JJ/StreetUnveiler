import os
import argparse
import numpy as np
from PIL import Image
import torch
import lpips
from tqdm import tqdm
import torchvision.transforms.functional as tf

def calculate_lpips(eval_path, gt_path):
    loss_fn = lpips.LPIPS(net='vgg').cuda()

    eval_images = sorted(os.listdir(eval_path))
    gt_images = sorted(os.listdir(gt_path))

    assert len(eval_images) == len(gt_images), "The number of images in both directories must be the same."

    lpips_values = []

    for eval_image, gt_image in tqdm(zip(eval_images, gt_images)):
        eval_img = Image.open(os.path.join(eval_path, eval_image))
        gt_img = Image.open(os.path.join(gt_path, gt_image))

        eval_tensor = tf.to_tensor(eval_img).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()

        lpips_value = loss_fn.forward(eval_tensor, gt_tensor)
        lpips_values.append(lpips_value.item())

    return lpips_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate LPIPS')
    parser.add_argument('--eval_path', type=str, required=True, help='Path to the directory containing evaluation images.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the directory containing ground truth images.')
    args = parser.parse_args()

    lpips_values = calculate_lpips(args.eval_path, args.gt_path)
    print(f'Average LPIPS: {np.mean(lpips_values)}')
