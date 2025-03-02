
#
# Written by: Jingwei Xu, ShanghaiTech University
# Based on the repo: https://github.com/ewrfcas/ZITS-PlusPlus
#

import argparse
import os
import sys

import cv2
import yaml
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import torch.nn.functional as FF
import torchvision.transforms.functional as F
import torch.nn.parallel
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../3rd_party/ZITS-PlusPlus')))

# import utils
from base.parse_config import ConfigParser
from dnnlib.util import get_obj_by_name
from trainers.nms_temp import get_nms as get_np_nms
from trainers.pl_trainers import wf_inference_test

def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))

def resize(img, height, width, center_crop=False):
    imgh, imgw = img.shape[0:2]

    if center_crop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    if imgh > height and imgw > width:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_LINEAR
    img = cv2.resize(img, (width, height), interpolation=inter)

    return img

ones_filter = np.ones((3, 3), dtype=np.float32)
d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)

def load_masked_position_encoding(mask):
    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1. - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (256 / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * 128).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, 128 - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct

def to_tensor(img, norm=False):
    # img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    if norm:
        img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img_t

class ZitsGuidance():
    def __init__(self):
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('--config', default="config.yml", type=str, help='config file path')
        args.add_argument('--exp_name', default=None, type=str, help='method name')
        args.add_argument('--dynamic_size', action='store_true', help='Whether to finetune in dynamic size?')
        args.add_argument('--use_ema', action='store_true', help='Whether to use ema?')
        args.add_argument('--ckpt_resume', default='last.ckpt', type=str, help='PL path to restore')
        args.add_argument('--wf_ckpt', type=str, default=None, help='Line detector weights')
        args.add_argument('--save_path', default='outputs', type=str, help='path to save')
        args.add_argument('--test_size', type=int, default=None, help='Test image size')
        args.add_argument('--binary_threshold', type=int, default=50, help='binary_threshold for E-NMS (from 0 to 255)')
        args.add_argument('--eval', action='store_true', help='Whether to eval?')
        args.add_argument('--save_image_only', action='store_true', help='Only save image')
        args.add_argument('--obj_removal', action='store_true', help='obj_removal')
        args.add_argument('--eval_path', type=str, default=None, help='Eval gt image path')
        args = args.parse_args(args=[
            '--config', os.path.join(os.getcwd(), '3rd_party/ZITS-PlusPlus/configs/config_zitspp_finetune.yml'),
            '--exp_name', 'model_512',
            '--ckpt_resume', os.path.join(os.getcwd(), '3rd_party/ZITS-PlusPlus/ckpts/model_512/models/last.ckpt'),
            '--wf_ckpt', os.path.join(os.getcwd(), '3rd_party/ZITS-PlusPlus/ckpts/best_lsm_hawp.pth'),
            '--use_ema',
            '--test_size', '512',
            '--save_image_only',
            '--obj_removal'
        ])
        self.args = args
        self.args.resume = None
        self.config = ConfigParser.from_args(args, mkdir=False)

        # build models architecture, then print to console
        structure_upsample = get_obj_by_name(self.config['structure_upsample_class'])()

        edgeline_tsr = get_obj_by_name(self.config['edgeline_tsr_class'])()
        grad_tsr = get_obj_by_name(self.config['grad_tsr_class'])()
        ftr = get_obj_by_name(self.config['g_class'])(config=self.config['g_args'])
        D = get_obj_by_name(self.config['d_class'])(config=self.config['d_args'])

        if 'PLTrainer' not in self.config.config or self.config['PLTrainer'] is None:
            self.config.config['PLTrainer'] = 'trainers.pl_trainers.FinetunePLTrainer'

        self.model = get_obj_by_name(self.config['PLTrainer'])(structure_upsample, edgeline_tsr, grad_tsr, ftr, D, self.config,
                                                     '3rd_party/ZITS-PlusPlus/ckpts/' + self.args.exp_name, use_ema=self.args.use_ema, dynamic_size=self.args.dynamic_size, test_only=True)

        if self.args.use_ema:
            self.model.reset_ema()

        if self.args.ckpt_resume:
            print("Loading checkpoint: {} ...".format(self.args.ckpt_resume))
            checkpoint = torch.load(self.args.ckpt_resume, map_location='cpu')
            torch_init_model(self.model, checkpoint, key='state_dict')

        if hasattr(self.model, "wf"):
            self.model.wf.load_state_dict(torch.load(self.args.wf_ckpt, map_location='cpu')['model'])

        self.model.cuda()

        if self.args.use_ema:
            self.model.ftr_ema.eval()
        else:
            self.model.ftr.eval()

    def get_batch(self, img_path, mask_path):
        img = cv2.imread(img_path)
        original_imgh, original_imgw, _ = img.shape
        if self.args.test_size is not None:
            img = resize(img, self.args.test_size, self.args.test_size)
        img = img[:, :, ::-1]
        # resize/crop if needed
        imgh, imgw, _ = img.shape
        img_512 = resize(img, 512, 512)
        img_256 = resize(img, 256, 256)

        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255
        mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_512 = (mask_512 > 127).astype(np.uint8) * 255

        batch = dict()
        batch['image'] = to_tensor(img.copy(), norm=True)[None, ...]
        batch['img_256'] = to_tensor(img_256, norm=True)[None, ...]
        batch['mask'] = to_tensor(mask)[None, ...]
        batch['mask_256'] = to_tensor(mask_256)[None, ...]
        batch['mask_512'] = to_tensor(mask_512)[None, ...]
        batch['img_512'] = to_tensor(img_512)[None, ...]
        batch['imgh'] = torch.tensor([imgh])[None, ...]
        batch['imgw'] = torch.tensor([imgw])[None, ...]
        batch['original_imgh'] = original_imgh
        batch['original_imgw'] = original_imgw

        batch['name'] = os.path.basename(img_path)

        # load pos encoding
        rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
        batch['rel_pos'] = torch.LongTensor(rel_pos)[None, ...]
        batch['abs_pos'] = torch.LongTensor(abs_pos)[None, ...]
        batch['direct'] = torch.LongTensor(direct)[None, ...]

        # load gradient
        if self.config['g_args']['use_gradient']:
            img_gray = rgb2gray(img_256) * 255
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float32)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float32)

            img_gray = rgb2gray(img) * 255
            sobelx_hr = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float32)
            sobely_hr = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float32)

            batch['gradientx'] = torch.from_numpy(sobelx).unsqueeze(0).float()[None, ...]
            batch['gradienty'] = torch.from_numpy(sobely).unsqueeze(0).float()[None, ...]
            batch['gradientx_hr'] = torch.from_numpy(sobelx_hr).unsqueeze(0).float()[None, ...]
            batch['gradienty_hr'] = torch.from_numpy(sobely_hr).unsqueeze(0).float()[None, ...]

        return batch


    def inpaint(self, img_path, mask_path, output_path):
        SEED = 123456
        torch.manual_seed(SEED)

        with torch.no_grad():
            batch = self.get_batch(img_path, mask_path)
            batch['size_ratio'] = -1
            batch['H'] = -1
            for k in batch:
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].cuda()

            # load line
            batch['line_256'] = wf_inference_test(self.model.wf, batch['img_512'], h=256, w=256, masks=batch['mask_512'],
                                                  valid_th=0.85, mask_th=0.85, obj_remove=self.args.obj_removal)
            imgh = batch['imgh'][0].item()
            imgw = batch['imgw'][0].item()

            # inapint prior
            edge_pred, line_pred = self.model.edgeline_tsr.forward(batch['img_256'], batch['line_256'], masks=batch['mask_256'])
            line_pred = batch['line_256'] * (1 - batch['mask_256']) + line_pred * batch['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            current_size = 256
            if current_size != min(imgh, imgw):
                while current_size * 2 <= max(imgh, imgw):
                    # nms for HR
                    line_pred = self.model.structure_upsample(line_pred)[0]
                    edge_pred_nms = get_np_nms(edge_pred, binary_threshold=self.args.binary_threshold)
                    edge_pred_nms = self.model.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                    current_size *= 2

                edge_pred_nms = FF.interpolate(edge_pred_nms, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
            else:
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

                if self.config['g_args']['use_gradient'] is True:
                    gradientx, gradienty = self.model.grad_tsr.forward(batch['img_256'], batch['gradientx'], batch['gradienty'], masks=batch['mask_256'])
                    gradientx = batch['gradientx'] * (1 - batch['mask_256']) + gradientx * batch['mask_256']
                    gradienty = batch['gradienty'] * (1 - batch['mask_256']) + gradienty * batch['mask_256']
                    gradientx = FF.interpolate(gradientx, size=(imgh, imgw), mode='bilinear')
                    gradientx = gradientx * batch['mask'] + batch['gradientx_hr'] * (1 - batch['mask'])

                    gradienty = FF.interpolate(gradienty, size=(imgh, imgw), mode='bilinear')
                    gradienty = gradienty * batch['mask'] + batch['gradienty_hr'] * (1 - batch['mask'])

                    batch['gradientx'] = gradientx.detach()
                    batch['gradienty'] = gradienty.detach()

            batch['edge'] = edge_pred.detach()
            batch['line'] = line_pred.detach()

            if self.args.use_ema:
                gen_ema_img, _ = self.model.run_G_ema(batch)
            else:
                gen_ema_img, _ = self.model.run_G(batch)
            gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
            gen_ema_img = (gen_ema_img + 1) / 2
            gen_ema_img = gen_ema_img * 255.0
            gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
            if not self.args.save_image_only:
                edge_res = (batch['edge'] * 255.0).permute(0, 2, 3, 1).int().cpu().numpy()
                edge_res = np.tile(edge_res, [1, 1, 1, 3])
                line_res = (batch['line'] * 255.0).permute(0, 2, 3, 1).int().cpu().numpy()
                line_res = np.tile(line_res, [1, 1, 1, 3])
                masked_img = (batch['image'] * (1 - batch['mask']) + 1) / 2 * 255
                masked_img = masked_img.permute(0, 2, 3, 1).int().cpu().numpy()
                if self.config['g_args']['use_gradient'] is True:
                    gradientx = (gradientx - gradientx.min()) / (gradientx.max() - gradientx.min() + 1e-7)
                    gradientx = np.tile((gradientx * 255.0).permute(0, 2, 3, 1).int().cpu().numpy(), [1, 1, 1, 3])
                    gradienty = (gradienty - gradienty.min()) / (gradienty.max() - gradienty.min() + 1e-7)
                    gradienty = np.tile((gradienty * 255.0).permute(0, 2, 3, 1).int().cpu().numpy(), [1, 1, 1, 3])
                    final_res = np.concatenate([masked_img, edge_res, line_res, gradientx, gradienty, gen_ema_img], axis=2)
                else:
                    final_res = np.concatenate([masked_img, edge_res, line_res, gen_ema_img], axis=2)
            else:
                final_res = gen_ema_img
            cv2.imwrite(output_path, cv2.resize(final_res[0, :, :, ::-1].astype(np.uint8), ( batch['original_imgw'], batch['original_imgh']), interpolation=cv2.INTER_AREA))
            # cv2.imwrite(output_path, final_res[0, :, :, ::-1])


if __name__ == '__main__':

    zits = ZitsGuidance()
    zits.inpaint(
        "",
        "",
        "",
    )