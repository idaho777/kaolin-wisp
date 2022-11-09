# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Callable
import torch
from torch.utils.data import Dataset
from wisp.datasets.formats import load_nerf_standard_data, load_rtmv_data
from wisp.core import Rays


class MultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.

    TODO(ttakikawa): Support single-camera dynamic temporal scenes, and multi-camera dynamic temporal scenes.
    TODO(ttakikawa): Currently this class only supports sampling per image, not sampling across the entire
                     dataset. This is due to practical reasons. Not sure if it matters...
    """

    def __init__(self, 
        dataset_path             : str,
        multiview_dataset_format : str      = 'standard',
        mip                      : int      = None,
        bg_color                 : str      = None,
        dataset_num_workers      : int      = -1,
        transform                : Callable = None,
        **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want 
        to load the images unless we have to. This might change later.

        Args: 
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.multiview_dataset_format = multiview_dataset_format
        self.mip = mip
        self.bg_color = bg_color
        self.dataset_num_workers = dataset_num_workers
        self.transform = transform

    def init(self):
        """Initializes the dataset.
        """

        # Get image tensors 
        
        self.coords = None

        self.data = self.get_images()
        B, H, W, C = self.data["imgs"].shape

        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]
        self.data["gradients"] = self.sobel_filter(self.data["imgs"].reshape(B, C, H, W))
        self.data["gradients"] = self.data["gradients"].reshape(B, H, W, C)

        self.data["imgs"] = self.data["imgs"].reshape(self.num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(self.num_imgs, -1, 3)
        self.data["gradients"] = self.data["gradients"].reshape(self.num_imgs, -1, 3)
        if "depths" in self.data:
            self.data["depths"] = self.data["depths"].reshape(self.num_imgs, -1, 1)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(self.num_imgs, -1, 1)
        # if "gradients" in self.data:
        

    def sobel_filter(self, img:torch.Tensor):
        """
        x: image to sobel over [batch, C, H, W]
        """
        batch, C, H, W = img.shape
        # if not isinstance(input, torch.Tensor):
        #     raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        # pix_img = x.numpy()
        # x_grad = ndimage.sobel(pix_img, axis=1)
        # y_grad = ndimage.sobel(pix_img, axis=0)
        # gpix_img = np.hypot(y_grad, x_grad) / (4* 2**2)

        blur = torch.tensor([-1, 0, 1]) / 2
        edge = torch.tensor([1, 2, 1]) / 4
        dx_f = blur[None, ...] * edge[..., None]
        dy_f = blur[..., None] * edge[None, ...]
        dx_f = dx_f.unsqueeze(0).unsqueeze(0).repeat((C, C, 1, 1))
        dy_f = dy_f.unsqueeze(0).unsqueeze(0).repeat((C, C, 1, 1))

        assert dx_f.shape == dy_f.shape == (C, C, 3, 3)

        # Call sobel filters
        dx = torch.conv2d(input=img, weight=dx_f, padding=1)
        dy = torch.conv2d(input=img, weight=dy_f, padding=1)

        assert dx.shape == dy.shape == (batch, C, H, W)
        grad = (dx**2 + dy**2).sqrt()
        grad = grad.reshape(batch, C, H, W)

        assert grad.shape == (batch, C, H, W)

        return grad

    def get_images(self, split='train', mip=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        if mip is None:
            mip = self.mip
        
        if self.multiview_dataset_format == "standard":
            data = load_nerf_standard_data(self.root, split,
                                            bg_color=self.bg_color, num_workers=self.dataset_num_workers, mip=self.mip)
            
        elif self.multiview_dataset_format == "rtmv":
            if split == 'train':
                data = load_rtmv_data(self.root, split,
                                      return_pointcloud=True, mip=mip, bg_color=self.bg_color,
                                      normalize=True, num_workers=self.dataset_num_workers)
                self.coords = data["coords"]
                self.coords_center = data["coords_center"]
                self.coords_scale = data["coords_scale"]
            else:
                if self.coords is None:
                    assert False and "Initialize the dataset first with the training data!"
                
                data = load_rtmv_data(self.root, split,
                                      return_pointcloud=False, mip=mip, bg_color=self.bg_color,
                                      normalize=False)
                
                data["depths"] = data["depths"] * self.coords_scale
                data["rays"].origins = (data["rays"].origins - self.coords_center) * self.coords_scale

        return data

    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns a ray.
        """
        out = {}
        out['rays'] = self.data["rays"][idx]
        out['imgs'] = self.data["imgs"][idx]

        if self.transform is not None:
            out = self.transform(out)
        
        return out
    
    def get_img_samples(self, idx, batch_size):
        """Returns a batch of samples from an image, indexed by idx.
        """

        ray_idx = torch.randperm(self.data["imgs"].shape[1])[:batch_size]

        out = {}
        out['rays'] = Rays(origins=self.data["rays"].origins[idx, ray_idx],
                dirs=self.data["rays"].dirs[idx, ray_idx],
                dist_min=self.data["rays"].dist_min, dist_max=self.data["rays"].dist_max)
        out['imgs'] = self.data["imgs"][idx, ray_idx]
        
        return out
