
#
# Written by: Jingwei Xu, ShanghaiTech University
# Based on the repo: https://github.com/ewrfcas/LeftRefill
#

import os
from glob import glob
import random
import sys

import numpy as np
import torch
from PIL import Image
from einops import repeat
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../3rd_party')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../3rd_party/LeftRefill')))

from LeftRefill.ldm.models.diffusion.ddim import DDIMSampler
from LeftRefill.ldm.util import instantiate_from_config


class LeftRefillGuidance():
    def __init__(self):
        # torch.set_grad_enabled(False)
        self.target_image_size = 512
        self.root_path = "3rd_party/LeftRefill/check_points/ref_guided_inpainting"
        self.repeat_sp_token = 50
        self.sp_token = "<special-token>"

        self.sampler = self.initialize_model(path=self.root_path)

    @torch.no_grad()
    def load_state_dict(self, ckpt_path, location='cpu'):
        def get_state_dict(d):
            return d.get('state_dict', d)

        _, extension = os.path.splitext(ckpt_path)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(ckpt_path, device=location)
        else:
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
        state_dict = get_state_dict(state_dict)
        # print(f'Loaded state_dict from [{ckpt_path}]')
        return state_dict


    @torch.no_grad()
    def torch_init_model(self, model, total_dict, key, rank=0):
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

        # if rank == 0:
        #     print("missing keys:{}".format(missing_keys))
        #     print('unexpected keys:{}'.format(unexpected_keys))
        #     print('error msgs:{}'.format(error_msgs))


    @torch.no_grad()
    def initialize_model(self, path):
        config = OmegaConf.load(os.path.join(path, "model_config.yaml"))
        model = instantiate_from_config(config.model)
        # self.repeat_sp_token = config['model']['params']['data_config']['self.repeat_sp_token']
        # self.sp_token = config['model']['params']['data_config']['self.sp_token']

        ckpt_list = glob(os.path.join(path, 'ckpts/epoch=*.ckpt'))
        if len(ckpt_list) > 1:
            resume_path = sorted(ckpt_list, key=lambda x: int(x.split('/')[-1].split('.ckpt')[0].split('=')[-1]))[-1]
        else:
            resume_path = ckpt_list[0]
        # print('Load ckpt', resume_path)

        reload_weights = self.load_state_dict(resume_path, location='cpu')
        self.torch_init_model(model, reload_weights, key='none')
        if getattr(model, 'save_prompt_only', False):
            pretrained_weights = self.load_state_dict('3rd_party/LeftRefill/pretrained_models/512-inpainting-ema.ckpt', location='cpu')
            self.torch_init_model(model, pretrained_weights, key='none')

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
        self.sampler = DDIMSampler(model)

        return self.sampler


    @torch.no_grad()
    def make_batch_sd(
            self,
            image,
            mask,
            txt,
            device,
            num_samples=1):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
        }
        return batch


    def inpaint(self, sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = self.sampler.model

        # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, 4, h // 8, w // 8)
        start_code = torch.from_numpy(start_code).to(
            device=device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda"):
            batch = self.make_batch_sd(image, mask, txt=prompt,
                                  device=device, num_samples=num_samples)
            # print(batch['image'].shape)
            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(
                        model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples)
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = self.sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)
            pred = x_samples_ddim * batch['mask'] + batch['image'] * (1 - batch['mask'])

            result = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)

            result = (result.cpu().numpy().transpose(0, 2, 3, 1) * 255)
            result = result[:, :, 512:]

        return [Image.fromarray(img.astype(np.uint8)) for img in result]
        # return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


    @torch.no_grad()
    def pad_image(self, input_image):
        pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
        im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
        return im_padded


    @torch.no_grad()
    def predict_default(self, source, mask, reference, ddim_steps = 50, num_samples = 1, scale = 2.5, seed = random.randint(0, 2147483647)):
        # source_img = source["image"].convert("RGB")
        source_img = source
        origin_w, origin_h = source_img.size
        ratio = origin_h / origin_w
        # init_mask = source["mask"].convert("RGB")
        init_mask = mask
        # print('Source...', source_img.size)
        # reference_img = reference.convert("RGB")
        reference_img = reference
        # print('Reference...', reference_img.size)
        # if min(width, height) > image_size_limit:
        #     if width > height:
        #         init_image = init_image.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.BICUBIC)
        #         init_mask = init_mask.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.LINEAR)
        #     else:
        #         init_image = init_image.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.BICUBIC)
        #         init_mask = init_mask.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.LINEAR)
        #     init_mask = np.array(init_mask)
        #     init_mask[init_mask > 0] = 255
        #     init_mask = Image.fromarray(init_mask)

        # directly resizing to 512x512
        source_img = source_img.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BICUBIC)
        reference_img = reference_img.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BICUBIC)
        init_mask = init_mask.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BILINEAR)
        init_mask = np.array(init_mask)
        init_mask[init_mask > 0] = 255
        init_mask = Image.fromarray(init_mask)

        source_img = self.pad_image(source_img)  # resize to integer multiple of 32
        reference_img = self.pad_image(reference_img)
        mask = self.pad_image(init_mask)  # resize to integer multiple of 32
        width, height = source_img.size
        width *= 2
        # print("Inpainting...", width, height)
        # print("Prompt:", prompt)

        # get inputs
        image = np.concatenate([np.asarray(reference_img), np.asarray(source_img)], axis=1)
        image = Image.fromarray(image)
        mask = np.asarray(mask)
        mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
        mask = Image.fromarray(mask)

        prompt = ""
        for i in range(self.repeat_sp_token):
            prompt = prompt + self.sp_token.replace('>', f'{i}> ')
        prompt = prompt.strip()
        # print('Prompt:', prompt)

        result = self.inpaint(
            sampler=self.sampler,
            image=image,
            mask=mask,
            prompt=prompt,
            seed=seed,
            scale=scale,
            ddim_steps=ddim_steps,
            num_samples=num_samples,
            h=height, w=width
        )
        result = [r.resize((int(512 / ratio), 512), resample=Image.Resampling.BICUBIC) for r in result]
        for r in result:
            print(r.size)

        return result

    @torch.no_grad()
    def predict(self, source, mask, reference, ddim_steps = 50, num_samples = 1, scale = 2.5, seed = random.randint(0, 2147483647)):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # source_img = source["image"].convert("RGB")
        source_img = source
        origin_w, origin_h = source_img.size
        # init_mask = source["mask"].convert("RGB")
        init_mask = mask
        # print('Source...', source_img.size)
        # reference_img = reference.convert("RGB")
        reference_img = reference
        # print('Reference...', reference_img.size)
        # if min(width, height) > image_size_limit:
        #     if width > height:
        #         init_image = init_image.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.BICUBIC)
        #         init_mask = init_mask.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.LINEAR)
        #     else:
        #         init_image = init_image.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.BICUBIC)
        #         init_mask = init_mask.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.LINEAR)
        #     init_mask = np.array(init_mask)
        #     init_mask[init_mask > 0] = 255
        #     init_mask = Image.fromarray(init_mask)

        # directly resizing to 512x512
        source_img = source_img.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BICUBIC)
        reference_img = reference_img.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BICUBIC)
        init_mask = init_mask.resize((self.target_image_size, self.target_image_size), resample=Image.Resampling.BILINEAR)
        init_mask = np.array(init_mask)
        init_mask[init_mask > 0] = 255
        init_mask = Image.fromarray(init_mask)

        source_img = self.pad_image(source_img)  # resize to integer multiple of 32
        reference_img = self.pad_image(reference_img)
        mask = self.pad_image(init_mask)  # resize to integer multiple of 32
        width, height = source_img.size
        width *= 2
        # print("Inpainting...", width, height)
        # print("Prompt:", prompt)

        # get inputs
        image = np.concatenate([np.asarray(reference_img), np.asarray(source_img)], axis=1)
        image = Image.fromarray(image)
        mask = np.asarray(mask)
        mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
        mask = Image.fromarray(mask)

        prompt = ""
        for i in range(self.repeat_sp_token):
            prompt = prompt + self.sp_token.replace('>', f'{i}> ')
        prompt = prompt.strip()
        # print('Prompt:', prompt)

        result = self.inpaint(
            sampler=self.sampler,
            image=image,
            mask=mask,
            prompt=prompt,
            seed=seed,
            scale=scale,
            ddim_steps=ddim_steps,
            num_samples=num_samples,
            h=height, w=width
        )
        result = [r.resize((origin_w, origin_h), resample=Image.Resampling.BICUBIC) for r in result]
        # for r in result:
        #     print(r.size)

        return result



def test_left_refill(source_path, mask_path, reference_path):
    from diffusers.utils import load_image
    left_refill = LeftRefillGuidance()
    result= left_refill.predict(
        load_image(source_path),
        load_image(mask_path),
        load_image(reference_path),
    )

    return result[0]

if __name__ == "__main__":

    result = test_left_refill(
        "",
        "",
        "",
    )

    result.save("")
