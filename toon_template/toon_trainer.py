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
            rb = self.pipeline(rays=rays, lod_idx=None, channels=["rgb", "density"], extra_channels=["gradient"])
            # print(rb.rgb.shape, img_gts.shape)
            timer.check("inference")

            # RGB Loss
            #rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            # Come up with a better loss?
            # - Toon shading loss
            # - Color Palette Loss
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            rgb_loss = rgb_loss.mean()

            # Gradient loss
            # g_gt_img = self.sobel_filter(data['imgs'])
            # g_gt_img = (255*g_gt_img).cpu().numpy().astype(np.uint8)
            g_img = rb.rgb
            # print(g_gt_img.shape, g_img.shape)
            # g_gt_img = self.sobel_filter(img)

            # rgb_grads = lap.forward(rb.rgb)
            # gt_grads = lap.forward(img_gts)
            
            loss += self.extra_args["rgb_loss"] * rgb_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()
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


    def spatial_derivative(self, img:torch.Tensor):
        """
        Input:
            x: image to sobel over [batch, C, H, W]

        Output:
            grad: [batch, C, H, W]
        """
        batch, C, H, W = img.shape
        # if not isinstance(input, torch.Tensor):
        #     raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        # pix_img = x.numpy()
        # x_grad = ndimage.sobel(pix_img, axis=1)
        # y_grad = ndimage.sobel(pix_img, axis=0)
        # gpix_img = np.hypot(y_grad, x_grad) / (4* 2**2)
        # device = img.device

        # blur = torch.tensor([-1, 0, 1]) / 2
        # edge = torch.tensor([1, 2, 1]) / 4
        # dx_f = blur[None, ...] * edge[..., None]
        # dy_f = blur[..., None] * edge[None, ...]
        # dx_f = dx_f.unsqueeze(0).unsqueeze(0).repeat((C, C, 1, 1)).to(device)
        # dy_f = dy_f.unsqueeze(0).unsqueeze(0).repeat((C, C, 1, 1)).to(device)

        # assert dx_f.shape == dy_f.shape == (C, C, 3, 3)

        # # Call sobel filters
        # dx = torch.conv2d(input=img, weight=dx_f, padding=1)
        # dy = torch.conv2d(input=img, weight=dy_f, padding=1)

        # grad = (dx**2 + dy**2).sqrt()
        # grad = grad.reshape(batch, C, H, W)

        # assert grad.shape == (batch, C, H, W)


        spatial_derivatives = kornia.filters.spatial_gradient(img)
        dx = spatial_derivatives[:, :, 0]
        dy = spatial_derivatives[:, :, 1]
        assert dx.shape == dy.shape == (batch, C, H, W)
        gradient = (dx**2 + dy**2).sqrt()
        return gradient


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
                rb = self.renderer.render(self.pipeline, rays, lod_idx=lod_idx)
                rb = rb.reshape(*img.shape[:2], -1)
                
                rb.view = None
                rb.hit = None

                rb.gts = img.cuda()
                rb.err = (rb.gts[...,:3] - rb.rgb[...,:3])**2
                psnr_total += psnr(rb.rgb[...,:3], rb.gts[...,:3])
                lpips_total += lpips(rb.rgb[...,:3], rb.gts[...,:3], lpips_model)
                ssim_total += ssim(rb.rgb[...,:3], rb.gts[...,:3])
                
                exrdict = rb.reshape(*img.shape[:2], -1).cpu().exr_dict()
                
                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                # Image Gradients
                g_gt_img = grads[idx]
                g_gt_img = (255*g_gt_img).cpu().numpy().astype(np.uint8)

                g_img = rb.gradient

                # write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
                write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb.numpy())
                write_png(os.path.join(self.valid_log_dir, "grad-gt" + out_name + ".png"), g_gt_img) 

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
        data = self.dataset.get_images(split="test", mip=1)
        imgs = list(data["imgs"])
        B, H, W, C = data["imgs"].shape
        # grads = list(self.sobel_filter(data["imgs"].permute(B, C, H, W)).permute(B, H, W, C))
        grads = list(self.spatial_derivative(data["imgs"].permute((0, 3, 1, 2))).permute((0, 2, 3, 1)))

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        metric_dicts = []
        lods = list(range(self.pipeline.nef.num_lods))[::-1]
        for lod in lods:
            metric_dicts.append(self.evaluate_metrics(epoch, data["rays"], imgs, lod, f"lod{lod}", grads=grads))
        df = pd.DataFrame(metric_dicts)
        df['lod'] = lods
        df.to_csv(os.path.join(self.valid_log_dir, "lod.csv"))
