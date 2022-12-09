# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging as log
from tqdm import tqdm
import pandas as pd
import torch
from lpips import LPIPS
from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays

from scipy import ndimage
import numpy as np
import torch.nn as nn

import kornia



class ToonTrainer(BaseTrainer):
    """ A template trainer for optimizing a single neural radiance field object.
    This template merely serves as an example: users should override and modify according to their project requirements.
    """

    def pre_epoch(self, epoch):
        """For example, Override pre_epoch to support feature grid pruning at the beginning of epoch. """
        super().pre_epoch(epoch)   
        
        if self.extra_args["prune_every"] > -1 and epoch > 0 and epoch % self.extra_args["prune_every"] == 0:

            self.pipeline.nef.prune()
            self.init_optimizer()


    def init_log_dict(self):
        """Custom log dict. """
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['grad_loss'] = 0.0

    def step(self, epoch, n_iter, data):
        """Implement the optimization over image-space loss. """
        self.scene_state.optimization.iteration = n_iter

        timer = PerfTimer(activate=False, show_memory=False)

        # Map to device
        rays = data['rays'].to(self.device)
        img_gts = data['imgs'].to(self.device)
        img_gts = img_gts.reshape(-1, 3)
        timer.check("map to device")

        self.optimizer.zero_grad(set_to_none=True)
        timer.check("zero grad")

        loss = 0
        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays, lod_idx=None, channels=["rgb", "density"])
            timer.check("inference")

            # Color palette decoder
            rgb_labels = self.pipeline.nef.palette_decoder(rb.rgb[..., :3])
            target_labels = self.pipeline.nef.img_2_palette_labels(img_gts[..., :3], self.pipeline.nef.palette).squeeze(0)
            

            ce_loss = nn.CrossEntropyLoss()(rgb_labels, target_labels)
            # rgb_toon = self.pipeline.nef.palette_labels_2_img(torch.argmax(rgb_labels, dim=-1), self.pipeline.nef.palette)

            # RGB Loss
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            rgb_loss = rgb_loss.mean()

            loss += self.extra_args["rgb_loss"] * rgb_loss
            loss += self.extra_args["rgb_loss"] * ce_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()
            self.log_dict['rgb_loss'] += ce_loss.item()
            timer.check("loss")

        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += 1
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        timer.check("backward and step")

        
    def log_tb(self, epoch):
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])
        
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'Loss/{key}', self.log_dict[key], epoch)

        log.info(log_text)

        self.pipeline.eval()


    def evaluate_metrics(self, epoch, rays, imgs, lod_idx, name=None, grads=None):
        ray_os = list(rays.origins)
        ray_ds = list(rays.dirs)
        lpips_model = LPIPS(net='vgg').cuda()

        psnr_total = 0.0
        lpips_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, (img, ray_o, ray_d) in tqdm(enumerate(zip(imgs, ray_os, ray_ds))):
                
                rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                rays = rays.reshape(-1, 3)
                rays = rays.to('cuda')
                # rb = self.renderer.render(self.pipeline, rays)
                rb = self.renderer.render(self.pipeline, rays, lod_idx=lod_idx)
                rb = rb.reshape(*img.shape[:2], -1)
                
                rb.view = None
                rb.hit = None
                rb.gts = img.cuda()

                rgb_labels = self.pipeline.nef.palette_decoder(rb.rgb[..., :3])
                target_labels = self.pipeline.nef.img_2_palette_labels(rb.gts[..., :3], self.pipeline.nef.palette).squeeze(0)

                rgb_toon = self.pipeline.nef.palette_labels_2_img(torch.argmax(rgb_labels, dim=-1), self.pipeline.nef.palette)
                gt_toon = self.pipeline.nef.palette_labels_2_img(torch.argmax(target_labels, dim=-1), self.pipeline.nef.palette)

                rb.err = (gt_toon[...,:3] - rgb_toon[...,:3])**2
                psnr_total += psnr(rgb_toon[...,:3], gt_toon[...,:3])
                lpips_total += lpips(rgb_toon[...,:3], gt_toon[...,:3], lpips_model)
                ssim_total += ssim(rgb_toon[...,:3], gt_toon[...,:3])
                # rb.err = (rb.gts[...,:3] - rb.rgb[...,:3])**2
                # psnr_total += psnr(rb.rgb[...,:3], rb.gts[...,:3])
                # lpips_total += lpips(rb.rgb[...,:3], rb.gts[...,:3], lpips_model)
                # ssim_total += ssim(rb.rgb[...,:3], rb.gts[...,:3])
                
                exrdict = rb.reshape(*img.shape[:2], -1).cpu().exr_dict()
                
                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                # write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
                write_png(os.path.join(self.valid_log_dir, "label-" + out_name + ".png"), (255*rgb_toon).cpu().byte().numpy())
                write_png(os.path.join(self.valid_log_dir, "label-gt-" + out_name + ".png"), (255*gt_toon).cpu().byte().numpy())

                write_png(os.path.join(self.valid_log_dir, "reg-"+out_name + ".png"), (rb).cpu().image().byte().rgb.numpy())
                write_png(os.path.join(self.valid_log_dir, "reg-gt-" + out_name + ".png"), (255*rb.gts).cpu().byte().numpy())

        psnr_total /= len(imgs)
        lpips_total /= len(imgs)  
        ssim_total /= len(imgs)
                
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
        log.info(log_text)
 
        return {"psnr" : psnr_total, "lpips": lpips_total, "ssim": ssim_total}


    def validate(self, epoch=0):
        self.pipeline.eval()

        log.info("Beginning validation...")
        
        # data = self.dataset.get_images(split="val", mip=2)
        data = self.dataset.get_images(split="test", mip=2)
        imgs = list(data["imgs"])
        B, H, W, C = data["imgs"].shape

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        metric_dicts = []
        lods = list(range(self.pipeline.nef.num_lods))[::-1]
        # lods = lods[1:]
        for lod in lods:
            metric_dicts.append(self.evaluate_metrics(epoch, data["rays"], imgs, lod, f"lod{lod}"))
        df = pd.DataFrame(metric_dicts)
        df['lod'] = lods
        df.to_csv(os.path.join(self.valid_log_dir, "lod.csv"))
