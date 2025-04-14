import os
import argparse
import numpy as np
from PIL import Image
import torch
import lpips
from tqdm import tqdm
import torchvision.transforms.functional as tf

def calculate_lpips(eval_path, reference_path):
    loss_fn = lpips.LPIPS(net='vgg').cuda()

    eval_images = sorted(os.listdir(eval_path))
    reference_images = sorted(os.listdir(reference_path))

    assert len(eval_images) == len(reference_images), "The number of images in both directories must be the same."

    lpips_values = []

    for eval_image, reference_image in tqdm(zip(eval_images, reference_images)):
        eval_img = Image.open(os.path.join(eval_path, eval_image))
        reference_img = Image.open(os.path.join(reference_path, reference_image))

        eval_tensor = tf.to_tensor(eval_img).unsqueeze(0)[:, :3, :, :].cuda()
        reference_tensor = tf.to_tensor(reference_img).unsqueeze(0)[:, :3, :, :].cuda()

        lpips_value = loss_fn.forward(eval_tensor, reference_tensor)
        lpips_values.append(lpips_value.item())

    return lpips_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate LPIPS')
    parser.add_argument('--eval_path', type=str, required=True, help='Path to the directory containing evaluation images.')
    parser.add_argument('--reference_path', type=str, required=True, help='Path to the directory containing ground truth images.')
    args = parser.parse_args()

    lpips_values = calculate_lpips(args.eval_path, args.reference_path)
    print(f'Average LPIPS: {np.mean(lpips_values)}')
