import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import Unet
from monai.networks.layers.factories import Norm

from generative.networks.nets import DiffusionModelUNet

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-b9": (32, 64, 96, 256, 800),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class NeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
        fwd_renderer=None,
        sh=0,
        pe=0,
        backbone="efficientnet-b7",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.fov_depth = fov_depth
        self.pe = pe
        self.sh = sh
        self.fwd_renderer = fwd_renderer
        assert backbone in backbones.keys()

        self.net2d3d = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,  # Condition with straight/hidden view
            out_channels=self.fov_depth,
            num_channels=backbones[backbone],
            attention_levels=[False, False, False, True, True],
            norm_num_groups=8,
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )
        
        self.net3d3d = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=1, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH,
                dropout=0.5
            )
        )
        
    def forward(self, image2d, cameras, timesteps=None, resample=True, is_training=False):
        _device = image2d.device
        batch = image2d.shape[0]
        dtype = image2d.dtype
        if timesteps is None:
            timesteps = torch.zeros((batch,), device=_device).long()
        
        R = cameras.R
        T = torch.zeros_like(cameras.T.unsqueeze_(-1))
        
        mat = torch.cat([R, T], dim=-1)
        # rot = R
        mid = self.net2d3d(
            x=image2d,
            context=mat.reshape(batch, 1, -1),
            timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        inv = torch.cat([torch.inverse(R), -T], dim=-1)
        grd = F.affine_grid(inv, mid.size()).type(dtype)
        
        mid_resample = F.grid_sample(mid, grd)
        
        if is_training:
            # Randomly return out_resample or out_explicit
            rng = torch.rand(1).item()
            if rng > 0.5:
                out = self.net3d3d(mid)
                out = F.grid_sample(out, grd)
            else:
                out = self.net3d3d(mid_resample)
        else:
            out = self.net3d3d(mid_resample)
        return out, mid_resample